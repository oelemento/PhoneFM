//
//  HealthKitManager.swift
//  PhoneFM
//
//  Pulls 30 days of cardio-relevant HealthKit data + clinical records
//  for on-device model inference. All processing stays on-device.
//

import Foundation
import HealthKit
import Combine

@MainActor
final class HealthKitManager: ObservableObject {

    static let shared = HealthKitManager()
    private let store = HKHealthStore()

    @Published var authorizationGranted = false

    // ---- HK types we'll read ----
    private var readTypes: Set<HKObjectType> {
        var set: Set<HKObjectType> = []
        // Continuous vitals
        let qIDs: [HKQuantityTypeIdentifier] = [
            .heartRate, .heartRateVariabilitySDNN, .restingHeartRate, .walkingHeartRateAverage,
            .oxygenSaturation, .respiratoryRate, .bodyTemperature,
            .stepCount, .activeEnergyBurned, .appleExerciseTime, .vo2Max,
            .bloodPressureSystolic, .bloodPressureDiastolic,
            .bodyMass, .height,
        ]
        for id in qIDs {
            if let t = HKQuantityType.quantityType(forIdentifier: id) { set.insert(t) }
        }
        // Sleep + categorical
        if let sleep = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) { set.insert(sleep) }
        if let mind  = HKObjectType.categoryType(forIdentifier: .mindfulSession) { set.insert(mind) }
        // ECG (Apple Watch)
        set.insert(HKObjectType.electrocardiogramType())
        // Workouts
        set.insert(HKObjectType.workoutType())
        // Clinical records (Apple Health Records via FHIR)
        let clinicalIDs: [HKClinicalTypeIdentifier] = [
            .allergyRecord, .conditionRecord, .immunizationRecord,
            .labResultRecord, .medicationRecord, .procedureRecord, .vitalSignRecord,
        ]
        for id in clinicalIDs {
            if let t = HKObjectType.clinicalType(forIdentifier: id) { set.insert(t) }
        }
        return set
    }

    func requestAuthorization() async {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        do {
            try await store.requestAuthorization(toShare: [], read: readTypes)
            authorizationGranted = true
        } catch {
            print("HK auth failed: \(error)")
            authorizationGranted = false
        }
    }

    // ---- 30-day cardio data window assembly ----

    struct CardioWindow {
        let endDate: Date
        let heartRateSamples:      [HKQuantitySample]
        let hrvSamples:            [HKQuantitySample]
        let stepDailyTotals:       [(date: Date, steps: Double)]
        let sleepSegments:         [HKCategorySample]
        let oxygenSamples:         [HKQuantitySample]
        let ecgSamples:            [HKElectrocardiogram]
        let workouts:              [HKWorkout]
        let bpSystolicSamples:     [HKQuantitySample]
        let conditions:            [HKClinicalRecord]
        let medications:           [HKClinicalRecord]
        let labResults:            [HKClinicalRecord]
    }

    func fetchLast30Days() async throws -> CardioWindow {
        let end = Date()
        let start = Calendar.current.date(byAdding: .day, value: -30, to: end)!
        let pred = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictEndDate)

        async let hr        = quantitySamples(.heartRate, predicate: pred)
        async let hrv       = quantitySamples(.heartRateVariabilitySDNN, predicate: pred)
        async let steps     = dailyTotals(.stepCount, start: start, end: end)
        async let sleep     = categorySamples(.sleepAnalysis, predicate: pred)
        async let o2        = quantitySamples(.oxygenSaturation, predicate: pred)
        async let ecg       = electrocardiograms(predicate: pred)
        async let workouts  = self.workouts(predicate: pred)
        async let bp        = quantitySamples(.bloodPressureSystolic, predicate: pred)
        async let conditions  = clinicalRecords(.conditionRecord)
        async let medications = clinicalRecords(.medicationRecord)
        async let labResults  = clinicalRecords(.labResultRecord)

        return CardioWindow(
            endDate:               end,
            heartRateSamples:      try await hr,
            hrvSamples:            try await hrv,
            stepDailyTotals:       try await steps,
            sleepSegments:         try await sleep,
            oxygenSamples:         try await o2,
            ecgSamples:            try await ecg,
            workouts:              try await workouts,
            bpSystolicSamples:     try await bp,
            conditions:            try await conditions,
            medications:           try await medications,
            labResults:            try await labResults,
        )
    }

    // MARK: - thin wrappers around HKQuery

    private func quantitySamples(_ id: HKQuantityTypeIdentifier,
                                  predicate: NSPredicate) async throws -> [HKQuantitySample] {
        guard let t = HKQuantityType.quantityType(forIdentifier: id) else { return [] }
        return try await withCheckedThrowingContinuation { cont in
            let q = HKSampleQuery(sampleType: t, predicate: predicate, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, err in
                if let err = err { cont.resume(throwing: err); return }
                cont.resume(returning: (samples as? [HKQuantitySample]) ?? [])
            }
            store.execute(q)
        }
    }

    private func categorySamples(_ id: HKCategoryTypeIdentifier,
                                  predicate: NSPredicate) async throws -> [HKCategorySample] {
        guard let t = HKCategoryType.categoryType(forIdentifier: id) else { return [] }
        return try await withCheckedThrowingContinuation { cont in
            let q = HKSampleQuery(sampleType: t, predicate: predicate, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, err in
                if let err = err { cont.resume(throwing: err); return }
                cont.resume(returning: (samples as? [HKCategorySample]) ?? [])
            }
            store.execute(q)
        }
    }

    private func dailyTotals(_ id: HKQuantityTypeIdentifier,
                              start: Date, end: Date) async throws -> [(date: Date, steps: Double)] {
        guard let t = HKQuantityType.quantityType(forIdentifier: id) else { return [] }
        return try await withCheckedThrowingContinuation { cont in
            let anchor = Calendar.current.startOfDay(for: start)
            let interval = DateComponents(day: 1)
            let q = HKStatisticsCollectionQuery(
                quantityType: t,
                quantitySamplePredicate: nil,
                options: .cumulativeSum,
                anchorDate: anchor,
                intervalComponents: interval)
            q.initialResultsHandler = { _, results, err in
                if let err = err { cont.resume(throwing: err); return }
                var rows: [(date: Date, steps: Double)] = []
                results?.enumerateStatistics(from: start, to: end) { stat, _ in
                    let v = stat.sumQuantity()?.doubleValue(for: .count()) ?? 0
                    rows.append((date: stat.startDate, steps: v))
                }
                cont.resume(returning: rows)
            }
            store.execute(q)
        }
    }

    private func electrocardiograms(predicate: NSPredicate) async throws -> [HKElectrocardiogram] {
        let t = HKObjectType.electrocardiogramType()
        return try await withCheckedThrowingContinuation { cont in
            let q = HKSampleQuery(sampleType: t, predicate: predicate, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, err in
                if let err = err { cont.resume(throwing: err); return }
                cont.resume(returning: (samples as? [HKElectrocardiogram]) ?? [])
            }
            store.execute(q)
        }
    }

    private func workouts(predicate: NSPredicate) async throws -> [HKWorkout] {
        let t = HKObjectType.workoutType()
        return try await withCheckedThrowingContinuation { cont in
            let q = HKSampleQuery(sampleType: t, predicate: predicate, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, err in
                if let err = err { cont.resume(throwing: err); return }
                cont.resume(returning: (samples as? [HKWorkout]) ?? [])
            }
            store.execute(q)
        }
    }

    private func clinicalRecords(_ id: HKClinicalTypeIdentifier) async throws -> [HKClinicalRecord] {
        guard let t = HKObjectType.clinicalType(forIdentifier: id) else { return [] }
        return try await withCheckedThrowingContinuation { cont in
            let q = HKSampleQuery(sampleType: t, predicate: nil, limit: HKObjectQueryNoLimit,
                                  sortDescriptors: nil) { _, samples, err in
                if let err = err { cont.resume(throwing: err); return }
                cont.resume(returning: (samples as? [HKClinicalRecord]) ?? [])
            }
            store.execute(q)
        }
    }
}
