import SwiftUI

// =====================================================================
//  ANALYZING VIEW — live progress driven by the backend stream step
//  Backend steps: 1 received · 2 loading models · 3 inference ·
//  4 processing masks · 5 success
// =====================================================================

struct AnalyzingView: View {
    /// Current backend step (1–5).
    let step: Int

    @State private var rotation: Double = 0
    @State private var pulse = false
    @State private var dots = 0

    private let timer = Timer.publish(every: 0.45, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 30) {
            Spacer().frame(minHeight: 8)

            ZStack {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .stroke(Color.brand.opacity(0.10 - Double(i) * 0.025), lineWidth: 1)
                        .frame(width: CGFloat(84 + i * 40), height: CGFloat(84 + i * 40))
                }

                Circle()
                    .trim(from: 0, to: 0.28)
                    .stroke(
                        AngularGradient(colors: [Color.brand, Color.brand.opacity(0)], center: .center),
                        style: StrokeStyle(lineWidth: 3, lineCap: .round)
                    )
                    .frame(width: 84, height: 84)
                    .rotationEffect(.degrees(rotation))

                Circle()
                    .fill(Color.brand.opacity(pulse ? 0.16 : 0.06))
                    .frame(width: 52, height: 52)
                    .scaleEffect(pulse ? 1.1 : 0.9)

                Image(systemName: "eye")
                    .font(.system(size: 22, weight: .ultraLight))
                    .foregroundStyle(Color.brand)
            }
            .onAppear {
                withAnimation(.linear(duration: 1.8).repeatForever(autoreverses: false)) { rotation = 360 }
                withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { pulse = true }
            }

            HStack(spacing: 4) {
                Text("ANALIZOWANIE")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .tracking(4)
                    .foregroundStyle(Color.textPrimary)
                Text(String(repeating: ".", count: dots))
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundStyle(Color.brand)
                    .frame(width: 22, alignment: .leading)
            }

            VStack(spacing: 8) {
                ProgressStep(label: "Wgrywanie obrazu", done: step >= 2, active: step == 1)
                ProgressStep(label: "Detekcja tarczy wzrokowej", done: step >= 4, active: step == 2 || step == 3)
                ProgressStep(label: "Pomiar wskaźnika C/D", done: step >= 5, active: step == 4)
                ProgressStep(label: "Klasyfikacja AI", done: false, active: step == 5)
            }
            .padding(20)
            .glassCard()
            .padding(.horizontal, DS.screenPadding)

            Spacer().frame(minHeight: 8)
        }
        .onReceive(timer) { _ in dots = (dots + 1) % 4 }
    }
}

struct ProgressStep: View {
    let label: String
    var done = false
    var active = false

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(done ? Color.brand.opacity(0.2) : active ? Color.brand.opacity(0.10) : Color.surface2)
                    .frame(width: 22, height: 22)
                if done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Color.brand)
                } else if active {
                    Circle().fill(Color.brand).frame(width: 7, height: 7)
                }
            }
            .animation(.easeInOut(duration: 0.3), value: done)
            .animation(.easeInOut(duration: 0.3), value: active)

            Text(label)
                .font(.system(size: 13, weight: done ? .semibold : .regular, design: .rounded))
                .foregroundStyle(done ? Color.textPrimary : active ? Color.brand : Color.textTertiary)

            Spacer()
        }
    }
}
