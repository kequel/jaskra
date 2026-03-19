import SwiftUI

// Error Screen
struct ErrorView: View {
    let message: String
    @ObservedObject var vm: GlaucomaViewModel
    @State private var appeared = false

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            RadialGradient(
                colors: [Color.dangerRed.opacity(0.06), .clear],
                center: .center,
                startRadius: 0,
                endRadius: 300
            )
            .ignoresSafeArea()

            VStack(spacing: 28) {
                ZStack {
                    Circle()
                        .fill(Color.dangerRed.opacity(0.1))
                        .frame(width: 90, height: 90)
                    Image(systemName: "xmark.octagon")
                        .font(.system(size: 38, weight: .ultraLight))
                        .foregroundStyle(Color.dangerRed)
                }
                .scaleEffect(appeared ? 1 : 0.6)
                .opacity(appeared ? 1 : 0)

                VStack(spacing: 8) {
                    Text("BŁĄD POŁĄCZENIA")
                        .font(.system(size: 16, weight: .bold, design: .monospaced))
                        .tracking(3)
                        .foregroundStyle(Color.textPrimary)
                    Text(message)
                        .font(.system(size: 13))
                        .foregroundStyle(Color.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                .opacity(appeared ? 1 : 0)

                VStack(spacing: 12) {
                    Button {
                        vm.runAnalysis()
                    } label: {
                        Label("Spróbuj ponownie", systemImage: "arrow.clockwise")
                            .font(.system(size: 15, weight: .semibold, design: .monospaced))
                            .tracking(1)
                            .foregroundStyle(Color.appBackground)
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(Color.accentCyan)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }

                    Button {
                        vm.reset()
                    } label: {
                        Text("Wróć do początku")
                            .font(.system(size: 14))
                            .foregroundStyle(Color.textSecondary)
                    }
                }
                .padding(.horizontal, 32)
                .opacity(appeared ? 1 : 0)
            }
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) { appeared = true }
        }
    }
}
