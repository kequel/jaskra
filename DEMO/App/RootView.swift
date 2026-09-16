import SwiftUI

// =====================================================================
//  ROOT VIEW — splash → (auth | main tabs)
// =====================================================================

struct RootView: View {
    @EnvironmentObject private var auth: AuthManager
    @State private var showSplash = true

    var body: some View {
        ZStack {
            if showSplash {
                SplashView()
                    .transition(.opacity)
            } else if auth.hasAccess {
                MainTabView()
                    .transition(.opacity)
            } else {
                AuthView()
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.45), value: showSplash)
        .animation(.easeInOut(duration: 0.45), value: auth.hasAccess)
        .task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            showSplash = false
        }
    }
}
