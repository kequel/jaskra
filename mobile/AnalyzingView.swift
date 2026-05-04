import SwiftUI

// Backend steps (from /analyze-glaucoma-stream):
//  step 1 – image received
//  step 2 – loading AI models (only on cold start)
//  step 3 – running AI inference
//  step 4 – processing masks
//  step 5 – success (final result)

// Analyzing Screen
struct AnalyzingView: View {
    @ObservedObject var vm: GlaucomaViewModel

    @State private var rotation: Double = 0
    @State private var pulse: Bool = false
    @State private var dots: Int = 0

    let timer = Timer.publish(every: 0.45, on: .main, in: .common).autoconnect()

    // Map backend step → which UI steps are "done" (steps 1-4 in UI)
    // UI step 1: Wgrywanie obrazu       → backend step >= 2
    // UI step 2: Detekcja tarczy        → backend step >= 3
    // UI step 3: Pomiar wskaźnika C/D   → backend step >= 4
    // UI step 4: Klasyfikacja AI        → backend step >= 5
    private func isDone(_ uiStep: Int) -> Bool {
        vm.analysisStep >= uiStep + 1
    }

    private func isActive(_ uiStep: Int) -> Bool {
        vm.analysisStep == uiStep
    }

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
                    // outer rings
                    ForEach(0..<3, id: \.self) { i in
                        Circle()
                            .stroke(Color.accentCyan.opacity(0.08 - Double(i) * 0.02), lineWidth: 1)
                            .frame(width: CGFloat(100 + i * 50), height: CGFloat(100 + i * 50))
                    }

                    // spinning arc
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

                    // pulse glow
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

                // text header
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

                    Text(stepDescription(vm.analysisStep))
                        .font(.system(size: 13))
                        .foregroundStyle(Color.textSecondary)
                        .animation(.easeInOut(duration: 0.3), value: vm.analysisStep)
                }

                // Progress steps — driven by real backend steps
                VStack(spacing: 10) {
                    ProgressStep(
                        label: "Wgrywanie obrazu",
                        done: isDone(1),
                        active: isActive(1)
                    )
                    ProgressStep(
                        label: "Detekcja tarczy wzrokowej",
                        done: isDone(2),
                        active: isActive(2) || isActive(3)   // steps 2+3 both "run AI"
                    )
                    ProgressStep(
                        label: "Pomiar wskaźnika C/D",
                        done: isDone(3),
                        active: isActive(4)
                    )
                    ProgressStep(
                        label: "Klasyfikacja AI",
                        done: isDone(4),
                        active: isActive(5)
                    )
                }
                .padding(.horizontal, 48)
            }
        }
        .onReceive(timer) { _ in
            dots = (dots + 1) % 4
        }
    }

    // Human-readable description for each backend step
    private func stepDescription(_ step: Int) -> String {
        switch step {
        case 0:  return "Łączenie z serwerem..."
        case 1:  return "Obraz odebrany — inicjalizacja modeli"
        case 2:  return "Ładowanie modeli AI (YOLO i UNet)"
        case 3:  return "Model AI przetwarza obraz dna oka"
        case 4:  return "Nakładanie masek segmentacji"
        case 5:  return "Analiza zakończona pomyślnie"
        default: return "Przetwarzanie..."
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
            .animation(.easeInOut(duration: 0.3), value: done)
            .animation(.easeInOut(duration: 0.3), value: active)

            Text(label)
                .font(.system(size: 12, weight: done ? .medium : .regular))
                .foregroundStyle(done ? Color.textPrimary : active ? Color.accentCyan.opacity(0.8) : Color.textTertiary)
                .animation(.easeInOut(duration: 0.3), value: done)
            Spacer()
        }
    }
}