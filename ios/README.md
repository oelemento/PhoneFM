# PhoneFM iOS app

Skeleton Swift files for the patient-facing iPhone app. These need an Xcode project to compile — SourceKit on the command line will flag missing types and macOS-only APIs because there is no `.xcodeproj` yet to set the iOS deployment target.

## One-time Xcode setup (Week 1, Day 1)

1. Xcode 16+ on a Mac running macOS 15+ (Apple Foundation Models framework needs the recent SDKs).
2. **File → New → Project → iOS → App** (SwiftUI, Swift, no Core Data, no tests).
   - Product name: `PhoneFM`
   - Organization identifier: `com.phonefm.app`
   - Interface: SwiftUI, Language: Swift
   - Save inside `ios/` and let it create `PhoneFM.xcodeproj` next to `PhoneFMApp/`.
3. Drag the existing files in `PhoneFMApp/` into the project navigator (Copy items if needed: OFF, Add to targets: ON).
4. In the target's **Signing & Capabilities** tab:
   - Add **HealthKit** capability → enable *Clinical Health Records*.
   - Add **Background Modes** → enable *Background processing*.
   - Add **Apple Foundation Models** capability (iOS 18+).
5. Replace the auto-generated `Info.plist` with `PhoneFMApp/Info.plist`.
6. Deployment target: iOS 18.0 (for FoundationModels).
7. Build & run on a physical iPhone (HealthKit and on-device LLM do not work in the simulator).

## Files

| File | Purpose |
|---|---|
| `PhoneFMApp.swift` | App entry point, BGTask scheduler registration |
| `ContentView.swift` | SwiftUI risk gauge + drivers + topic-for-doctor card |
| `HealthKitManager.swift` | Reads 30 days of HK quantity/category/clinical samples |
| `CardioFMInference.swift` | Core ML wrapper around the All-of-Us-trained model |
| `Info.plist` | HealthKit permission strings + background-task identifier |

## Model bundle

Drop the exported Core ML file at `PhoneFMApp/Resources/cardio_fm_v1.mlmodelc/` once Workstream A finishes Week 3. The inference wrapper looks for that filename.

## TestFlight beta (Week 4)

- Bundle identifier: `com.phonefm.app`
- Apple Developer account ($99/yr) needed for TestFlight distribution.
- First wave: 5–10 internal users (us + colleagues with iPhones).
