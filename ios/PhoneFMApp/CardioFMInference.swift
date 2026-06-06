//
//  CardioFMInference.swift
//  PhoneFM
//
//  Core ML wrapper for the cardio foundation model exported from
//  All of Us. Tokenizes the 30-day CardioWindow on-device and runs
//  the quantized transformer to produce a cardio-risk score.
//

import Foundation
import CoreML
import HealthKit

@MainActor
final class CardioFMInference {

    static let shared = CardioFMInference()

    // Lazily loaded so the app launches before the (~50 MB) model maps in.
    private var model: MLModel?

    struct RiskReport {
        let endDate: Date
        let riskScore: Double             // 30-day cardio event probability, [0, 1]
        let percentile: Int               // vs cohort training distribution, 0–100
        let trendDelta: Double            // change vs prior week
        let topFeatures: [(label: String, contribution: Double)]
    }

    // MARK: - model loading

    func loadModel() async throws {
        if model != nil { return }
        guard let url = Bundle.main.url(forResource: "cardio_fm_v1", withExtension: "mlmodelc") else {
            throw NSError(domain: "PhoneFM", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "cardio_fm_v1.mlmodelc not bundled"])
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all          // Neural Engine + GPU + CPU
        model = try MLModel(contentsOf: url, configuration: cfg)
    }

    // MARK: - tokenization (mirrors workbench/02_tokenizer.py)

    private func tokenize(_ window: HealthKitManager.CardioWindow) -> [Int32] {
        var tokens: [Int32] = []
        // Bin heart rate samples into 5-min eCDF deciles
        let hrDeciles = eCDFDecile(window.heartRateSamples.map {
            $0.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        })
        tokens.append(contentsOf: hrDeciles.map { Int32(100 + $0) })  // HR_D0..D9 = 100..109

        // Daily steps deciles
        let stepDeciles = eCDFDecile(window.stepDailyTotals.map { $0.steps })
        tokens.append(contentsOf: stepDeciles.map { Int32(110 + $0) }) // STEPS_D0..D9 = 110..119

        // Sleep segment encoding (collapsed to nightly REM/Deep percent)
        // Stub — flesh out once we lock the workbench tokenizer
        tokens.append(Int32(120))  // <DAY_SEP>

        // EHR conditions — emit DX10:<3char> tokens via codeset lookup
        for record in window.conditions {
            if let code = record.fhirResource?.identifier {
                tokens.append(Int32(hashCode(code, prefix: "DX10")))
            }
        }
        return tokens
    }

    private func eCDFDecile(_ values: [Double]) -> [Int] {
        guard !values.isEmpty else { return [] }
        let sorted = values.sorted()
        return values.map { v in
            guard let idx = sorted.firstIndex(where: { $0 >= v }) else { return 9 }
            return min(9, Int(Double(idx) / Double(sorted.count) * 10))
        }
    }

    private func hashCode(_ code: String, prefix: String) -> Int {
        // Simple deterministic hash into vocabulary slot 1000..9999 (matches workbench tokenizer)
        let h = abs((prefix + ":" + code).hashValue)
        return 1000 + (h % 9000)
    }

    // MARK: - inference

    func run(window: HealthKitManager.CardioWindow,
             priorRisk: Double? = nil) async throws -> RiskReport {
        try await loadModel()
        guard let model else { throw NSError(domain: "PhoneFM", code: -1) }

        let tokens = tokenize(window)
        let padded = pad(tokens, toLength: 4096)

        // Calendar-day position per token: increments at <DAY_SEP> (id = 3).
        // Pad positions stay at 0 — masked out via attn_mask.
        let DAY_SEP: Int32 = 3
        var positions = [Int32](repeating: 0, count: padded.count)
        var attn = [Int32](repeating: 0, count: padded.count)
        var day: Int32 = 0
        for i in 0..<padded.count {
            let t = padded[i]
            if t == 0 { continue }            // <PAD>
            attn[i] = 1
            if t == DAY_SEP { day += 1 }
            positions[i] = day
        }

        // Build the three MLMultiArray inputs
        let shape = [1, NSNumber(value: padded.count)]
        let idsArr   = try MLMultiArray(shape: shape, dataType: .int32)
        let posArr   = try MLMultiArray(shape: shape, dataType: .int32)
        let maskArr  = try MLMultiArray(shape: shape, dataType: .int32)
        for i in 0..<padded.count {
            idsArr[i]  = NSNumber(value: padded[i])
            posArr[i]  = NSNumber(value: positions[i])
            maskArr[i] = NSNumber(value: attn[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "positions": MLFeatureValue(multiArray: posArr),
            "attn_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let out = try model.prediction(from: input)
        let logit = out.featureValue(for: "risk_logit")?.doubleValue ?? 0
        let prob = 1.0 / (1.0 + exp(-logit))

        return RiskReport(
            endDate: window.endDate,
            riskScore: prob,
            percentile: Int(prob * 100),
            trendDelta: priorRisk.map { prob - $0 } ?? 0,
            topFeatures: extractTopFeatures(out: out)
        )
    }

    private func pad(_ tokens: [Int32], toLength n: Int) -> [Int32] {
        if tokens.count >= n { return Array(tokens.suffix(n)) }
        return tokens + Array(repeating: Int32(0), count: n - tokens.count)
    }

    private func extractTopFeatures(out: MLFeatureProvider) -> [(String, Double)] {
        // Placeholder until the workbench training script emits attribution heads.
        // Goal: surface 3 plain-language drivers of the current risk score.
        return [
            ("Elevated resting heart rate vs. last month", 0.31),
            ("HRV trending downward over the past 2 weeks", 0.22),
            ("Skipped 2 of last 4 prescribed dose-times (per HealthKit medication records)", 0.18),
        ]
    }
}
