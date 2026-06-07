import SwiftUI

// =====================================================================
//  SPLASH / LAUNCH SCREEN
//  Branded launch with a subtle reveal — no flashy gimmicks.
// =====================================================================

struct SplashView: View {
    @State private var appeared = false

    var body: some View {
        ZStack {
            ScreenBackground()

            VStack(spacing: 22) {
                ZStack {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(Color.brand.opacity(0.16))
                        .frame(width: 104, height: 104)
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(Color.brand.opacity(0.35), lineWidth: 1)
                        .frame(width: 104, height: 104)
                    Image(systemName: "eye.fill")
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(Color.brand)
                }
                .scaleEffect(appeared ? 1 : 0.82)
                .opacity(appeared ? 1 : 0)

                VStack(spacing: 8) {
                    Text("Jaskra")
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(Color.textPrimary)
                    Text("Diagnostyka dna oka")
                        .font(.system(size: 14, weight: .medium, design: .rounded))
                        .tracking(2)
                        .textCase(.uppercase)
                        .foregroundStyle(Color.textSecondary)
                }
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 10)
            }
        }
        .onAppear {
            withAnimation(.spring(response: 0.7, dampingFraction: 0.8)) { appeared = true }
        }
    }
}
