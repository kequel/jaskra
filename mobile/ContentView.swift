import SwiftUI

// Root Router
struct ContentView: View {
    @StateObject private var vm = GlaucomaViewModel()

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            switch vm.screen {
            case .home:
                HomeView(vm: vm)
                    .transition(.opacity)

            case .analyzing:
                AnalyzingView()
                    .transition(.opacity)

            case .result(let result):
                ResultView(vm: vm, result: result)
                    .transition(.move(edge: .bottom).combined(with: .opacity))

            case .error(let msg):
                ErrorView(message: msg, vm: vm)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: screenKey)
    }

    // helper for animation value
    private var screenKey: String {
        switch vm.screen {
        case .home:      return "home"
        case .analyzing: return "analyzing"
        case .result:    return "result"
        case .error:     return "error"
        }
    }
}

#Preview {
    ContentView()
}
