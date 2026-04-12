import SwiftUI

// Analyzing Screen
struct AnalyzingView: View {
    @State private var rotation: Double = 0
    @State private var pulse: Bool = false
    @State private var dots: Int = 0

    let timer = Timer.publish(every: 0.45, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            RadialGradient(
                colors: [Color.accentCyan.opacity(0.06), .clear],
                center: .center,
                startRadius: 0,
                endRadius: 280
            )
            .ignoresSafeArea()

            VStack(spacing: 40) {

                // circles animation
                ZStack {
                    // outer
                    ForEach(0..<3, id: \.self) { i in
                        Circle()
                            .stroke(Color.accentCyan.opacity(0.08 - Double(i) * 0.02), lineWidth: 1)
                            .frame(width: CGFloat(100 + i * 50), height: CGFloat(100 + i * 50))
                    }

                    // spinning
                    Circle()
                        .trim(from: 0, to: 0.28)
                        .stroke(
                            AngularGradient(
                                colors: [Color.accentCyan, Color.accentCyan.opacity(0)],
                                center: .center
                            ),
                            style: StrokeStyle(lineWidth: 2, lineCap: .round)
                        )
                        .frame(width: 100, height: 100)
                        .rotationEffect(.degrees(rotation))

                    // pulse
                    Circle()
                        .fill(Color.accentCyan.opacity(pulse ? 0.15 : 0.05))
                        .frame(width: 60, height: 60)
                        .scaleEffect(pulse ? 1.1 : 0.9)

                    // eye icon
                    Image(systemName: "eye")
                        .font(.system(size: 22, weight: .ultraLight))
                        .foregroundStyle(Color.accentCyan)
                }
                .onAppear {
                    withAnimation(.linear(duration: 1.8).repeatForever(autoreverses: false)) {
                        rotation = 360
                    }
                    withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
                        pulse = true
                    }
                }

                // text
                VStack(spacing: 10) {
                    HStack(spacing: 4) {
                        Text("ANALIZOWANIE")
                            .font(.system(size: 15, weight: .bold, design: .monospaced))
                            .tracking(4)
                            .foregroundStyle(Color.textPrimary)
                        Text(String(repeating: ".", count: dots))
                            .font(.system(size: 15, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color.accentCyan)
                            .frame(width: 24, alignment: .leading)
                    }

                    Text("Model AI przetwarza obraz dna oka")
                        .font(.system(size: 13))
                        .foregroundStyle(Color.textSecondary)
                }

                // TODO: Progress steps
                VStack(spacing: 10) {
                    ProgressStep(label: "Wgrywanie obrazu", done: true)
                    ProgressStep(label: "Detekcja tarczy wzrokowej", done: dots > 1)
                    ProgressStep(label: "Pomiar wskaźnika C/D", done: dots > 2)
                    ProgressStep(label: "Klasyfikacja AI", done: false, active: true)
                }
                .padding(.horizontal, 48)
            }
        }
        .onReceive(timer) { _ in
            dots = (dots + 1) % 4
        }
    }
}

struct ProgressStep: View {
    let label: String
    var done: Bool = false
    var active: Bool = false

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(done ? Color.accentCyan.opacity(0.2) : active ? Color.accentCyan.opacity(0.08) : Color.cardBorder)
                    .frame(width: 20, height: 20)
                if done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(Color.accentCyan)
                } else if active {
                    Circle()
                        .fill(Color.accentCyan)
                        .frame(width: 6, height: 6)
                }
            }
            Text(label)
                .font(.system(size: 12, weight: done ? .medium : .regular))
                .foregroundStyle(done ? Color.textPrimary : active ? Color.accentCyan.opacity(0.8) : Color.textTertiary)
            Spacer()
        }
    }
}
