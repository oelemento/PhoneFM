//
//  ContentView.swift
//  PhoneFM
//
//  Patient-facing UI:
//   - Weekly cardio-risk gauge + trend arrow
//   - 30-day sparkline
//   - 3 plain-language drivers (from CardioFMInference top features)
//   - "Topic for your doctor" card (Apple Foundation Models explanation)
//   - Disclaimer: not a diagnosis, not for emergency use.
//

import SwiftUI

struct ContentView: View {

    @EnvironmentObject private var hk: HealthKitManager
    @State private var report: CardioFMInference.RiskReport?
    @State private var loading = false
    @State private var errorText: String?
    @State private var explanation: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    if !hk.authorizationGranted {
                        permissionCard
                    } else if let report {
                        gaugeCard(report)
                        driversCard(report)
                        if let explanation { topicForDoctorCard(explanation) }
                    } else {
                        emptyCard
                    }

                    if let errorText {
                        Text(errorText).foregroundStyle(.red).font(.callout)
                    }

                    Spacer(minLength: 16)
                    disclaimerFooter
                }
                .padding()
            }
            .navigationTitle("Cardio risk trend")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(loading)
                }
            }
            .task { await refresh() }
        }
    }

    // MARK: - cards

    private var permissionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Connect your health data").font(.title2.bold())
            Text("PhoneFM reads your heart rate, HRV, sleep, activity, and connected health records on this iPhone. Nothing leaves your device.")
                .font(.callout)
            Button("Grant access") {
                Task { await hk.requestAuthorization() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.systemGray6)))
    }

    private func gaugeCard(_ r: CardioFMInference.RiskReport) -> some View {
        VStack(spacing: 8) {
            Text("This week").font(.headline)
            Gauge(value: r.riskScore, in: 0...1) {
                Text("30-day cardio risk")
            } currentValueLabel: {
                Text("\(r.percentile)").font(.system(size: 56, weight: .bold))
            }
            .gaugeStyle(.accessoryCircularCapacity)
            .tint(gradientForRisk(r.riskScore))
            .scaleEffect(1.8)
            .frame(height: 160)
            HStack {
                Image(systemName: r.trendDelta >= 0 ? "arrow.up.right" : "arrow.down.right")
                Text(String(format: "%+.0f points vs. last week", r.trendDelta * 100))
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.systemGray6)))
    }

    private func driversCard(_ r: CardioFMInference.RiskReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What's driving this").font(.headline)
            ForEach(r.topFeatures.indices, id: \.self) { i in
                let f = r.topFeatures[i]
                HStack(alignment: .top, spacing: 12) {
                    Circle().fill(.tint).frame(width: 8, height: 8).padding(.top, 8)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(f.label).font(.callout)
                        ProgressView(value: f.contribution)
                            .tint(.gray)
                            .frame(maxWidth: 140)
                    }
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.systemGray6)))
    }

    private func topicForDoctorCard(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Topic for your doctor", systemImage: "stethoscope")
                .font(.headline)
            Text(text).font(.callout)
            Button {
                let activity = UIActivityViewController(activityItems: [text], applicationActivities: nil)
                UIApplication.shared
                    .connectedScenes
                    .compactMap { $0 as? UIWindowScene }
                    .first?.windows.first?.rootViewController?
                    .present(activity, animated: true)
            } label: {
                Label("Share with your clinician", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color.accentColor.opacity(0.08)))
    }

    private var emptyCard: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text("Computing your risk trend…").font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 160)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.systemGray6)))
    }

    private var disclaimerFooter: some View {
        Text("PhoneFM is a research preview. It is not a medical device and does not diagnose, treat, or replace care from your clinician. In an emergency call 911.")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }

    private func gradientForRisk(_ p: Double) -> Gradient {
        switch p {
        case ..<0.2:   return Gradient(colors: [.green])
        case ..<0.5:   return Gradient(colors: [.green, .yellow])
        case ..<0.75:  return Gradient(colors: [.yellow, .orange])
        default:        return Gradient(colors: [.orange, .red])
        }
    }

    // MARK: - data

    @MainActor
    private func refresh() async {
        guard hk.authorizationGranted else { return }
        loading = true; errorText = nil
        do {
            let window = try await hk.fetchLast30Days()
            let r = try await CardioFMInference.shared.run(window: window)
            self.report = r
            self.explanation = await Self.composeExplanation(report: r)
        } catch {
            errorText = "Couldn't compute risk: \(error.localizedDescription)"
        }
        loading = false
    }

    /// Wrap the top features in a plain-language paragraph using Apple's
    /// on-device Foundation Models framework (iOS 18+). Until that SDK is
    /// available in this project, return a deterministic template.
    private static func composeExplanation(report: CardioFMInference.RiskReport) async -> String {
        let bullets = report.topFeatures.map { "• \($0.label)" }.joined(separator: "\n")
        return """
        Your 30-day cardio risk score is in the \(report.percentile)th percentile of users in your age band. The main contributors right now are:

        \(bullets)

        Worth bringing up at your next visit: ask whether the resting-HR and HRV trends warrant a clinical evaluation. This is not a diagnosis.
        """
    }
}

#Preview {
    ContentView()
        .environmentObject(HealthKitManager.shared)
}
