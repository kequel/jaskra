import SwiftUI

@main
struct GlaucomaApp: App {
    @StateObject private var theme = ThemeManager()
    @StateObject private var auth = AuthManager()
    @StateObject private var patients = PatientStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(theme)
                .environmentObject(auth)
                .environmentObject(patients)
                .tint(.brand)
                .preferredColorScheme(theme.colorScheme)
        }
    }
}
