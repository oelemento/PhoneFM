//
//  PhoneFMApp.swift
//  PhoneFM
//
//  App entry point. On launch: request HealthKit auth, schedule
//  background refresh, fall through to ContentView.
//

import SwiftUI
import BackgroundTasks

@main
struct PhoneFMApp: App {

    @StateObject private var hk = HealthKitManager.shared
    private let refreshTaskID = "com.phonefm.app.refresh"

    init() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: refreshTaskID, using: nil) { task in
            Task {
                await PhoneFMApp.runBackgroundRefresh()
                task.setTaskCompleted(success: true)
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(hk)
                .task {
                    if !hk.authorizationGranted { await hk.requestAuthorization() }
                    scheduleNextRefresh()
                }
        }
    }

    private func scheduleNextRefresh() {
        let req = BGProcessingTaskRequest(identifier: refreshTaskID)
        req.requiresNetworkConnectivity = false
        req.requiresExternalPower = false
        req.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60 * 24)  // ~24h cadence
        try? BGTaskScheduler.shared.submit(req)
    }

    static func runBackgroundRefresh() async {
        do {
            let window = try await HealthKitManager.shared.fetchLast30Days()
            _ = try await CardioFMInference.shared.run(window: window)
            // Persist the latest report to UserDefaults / SwiftData here.
        } catch {
            print("Background refresh failed: \(error)")
        }
    }
}
