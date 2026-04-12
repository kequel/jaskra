import SwiftUI

// Result Screen
struct ResultView: View {
    @ObservedObject var vm: GlaucomaViewModel
    let result: GlaucomaResult
    @State private var appeared = false

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            // Tinted glow based on diagnosis
            RadialGradient(
                colors: [
                    (result.hasGlaucoma ? Color.dangerRed : Color.positiveGreen).opacity(0.07),
                    .clear
                ],
                center: .top,
                startRadius: 0,
                endRadius: 400
            )
            .ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 24) {

                    //  Diagnosis Banner 
                    VStack(spacing: 16) {
                        ZStack {
                            Circle()
                                .fill((result.hasGlaucoma ? Color.dangerRed : Color.positiveGreen).opacity(0.12))
                                .frame(width: 90, height: 90)
                            Circle()
                                .stroke((result.hasGlaucoma ? Color.dangerRed : Color.positiveGreen).opacity(0.3), lineWidth: 1)
                                .frame(width: 90, height: 90)
                            Image(systemName: result.hasGlaucoma ? "exclamationmark.triangle.fill" : "checkmark.seal.fill")
                                .font(.system(size: 36, weight: .light))
                                .foregroundStyle(result.hasGlaucoma ? Color.dangerRed : Color.positiveGreen)
                        }
                        .padding(.top, 48)
                        .scaleEffect(appeared ? 1 : 0.7)
                        .opacity(appeared ? 1 : 0)

                        VStack(spacing: 6) {
                            Text(result.hasGlaucoma ? "WYKRYTO CECHY JASKRY" : "BRAK CECH JASKRY")
                                .font(.system(size: 20, weight: .bold, design: .monospaced))
                                .tracking(2)
                                .foregroundStyle(result.hasGlaucoma ? Color.dangerRed : Color.positiveGreen)

                            Text("Pewność modelu: \(Int(result.confidence * 100))%")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(Color.textSecondary)
                        }
                        .opacity(appeared ? 1 : 0)
                        .offset(y: appeared ? 0 : 10)
                    }
                    .animation(.spring(response: 0.6, dampingFraction: 0.7), value: appeared)

                    //  Processed Image 
                    if let img = vm.resultUIImage(from: result.imageBase64) {
                        ZStack(alignment: .bottomTrailing) {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .overlay(ScanLineEffect().clipShape(RoundedRectangle(cornerRadius: 16)))
                                .glowBorder(
                                    result.hasGlaucoma ? Color.dangerRed.opacity(0.5) : Color.positiveGreen.opacity(0.5),
                                    radius: 16
                                )

                            Label("AI Overlay", systemImage: "sparkles")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(.ultraThinMaterial, in: Capsule())
                                .padding(12)
                        }
                        .padding(.horizontal, 24)
                        .opacity(appeared ? 1 : 0)
                        .animation(.easeOut(duration: 0.5).delay(0.3), value: appeared)
                    }

                    // TODO: Metrics 
                    VStack(spacing: 12) {
                        MetricCard(
                            icon: "chart.pie.fill",
                            label: "Wskaźnik C/D (CDR)",
                            value: String(format: "%.2f", result.cupToDiscRatio),
                            color: vm.riskColor(cdr: result.cupToDiscRatio)
                        )
                        MetricCard(
                            icon: "waveform.path.ecg",
                            label: "Pewność klasyfikacji",
                            value: "\(Int(result.confidence * 100))%",
                            color: .accentCyan
                        )
                        MetricCard(
                            icon: "exclamationmark.shield.fill",
                            label: "Poziom ryzyka",
                            value: vm.riskText(cdr: result.cupToDiscRatio),
                            color: vm.riskColor(cdr: result.cupToDiscRatio)
                        )
                    }
                    .padding(.horizontal, 24)
                    .opacity(appeared ? 1 : 0)
                    .animation(.easeOut(duration: 0.5).delay(0.45), value: appeared)

                    // CDR Scale visual 
                    CDRScaleView(value: result.cupToDiscRatio)
                        .padding(.horizontal, 24)
                        .opacity(appeared ? 1 : 0)
                        .animation(.easeOut(duration: 0.5).delay(0.55), value: appeared)

                    //  Disclaimer 2
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(Color.textTertiary)
                            .font(.system(size: 14))
                            .padding(.top, 1)
                        Text("Wynik jest generowany przez model AI i ma charakter wyłącznie informacyjny. Skonsultuj się z okulistą w celu postawienia diagnozy.")
                            .font(.system(size: 12))
                            .foregroundStyle(Color.textTertiary)
                    }
                    .padding(.horizontal, 24)
                    .opacity(appeared ? 1 : 0)

                    // CTA Buttons 
                    VStack(spacing: 12) {
                        Button {
                            vm.selectedImage = nil
                            vm.screen = .home
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "plus.circle.fill")
                                Text("NOWA ANALIZA")
                                    .tracking(2)
                            }
                            .font(.system(size: 15, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color.appBackground)
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(Color.accentCyan)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                            .shadow(color: Color.accentCyan.opacity(0.3), radius: 12, y: 4)
                        }

                        Button {
                            vm.reset()
                        } label: {
                            Text("Wróć do początku")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(Color.textSecondary)
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.bottom, 48)
                    .opacity(appeared ? 1 : 0)
                    .animation(.easeOut(duration: 0.5).delay(0.6), value: appeared)
                }
            }
        }
        .onAppear { appeared = true }
    }
}

// CDR Scale
struct CDRScaleView: View {
    let value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("SKALA C/D")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(Color.textTertiary)

            ZStack(alignment: .leading) {
                // Track
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.cardBorder)
                    .frame(height: 8)

                // Gradient fill
                RoundedRectangle(cornerRadius: 4)
                    .fill(
                        LinearGradient(
                            colors: [.positiveGreen, .warningAmber, .dangerRed],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: CGFloat(value) * (UIScreen.main.bounds.width - 48 - 32), height: 8)

                // Thumb
                GeometryReader { geo in
                    Circle()
                        .fill(Color.white)
                        .frame(width: 16, height: 16)
                        .shadow(color: .black.opacity(0.3), radius: 4)
                        .offset(x: CGFloat(value) * (geo.size.width - 16), y: -4)
                }
            }
            .frame(height: 16)

            HStack {
                Text("0.0\nNiskie")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.positiveGreen.opacity(0.8))
                    .multilineTextAlignment(.center)
                Spacer()
                Text("0.5\nŚrednie")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.warningAmber.opacity(0.8))
                    .multilineTextAlignment(.center)
                Spacer()
                Text("0.8+\nWysokie")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.dangerRed.opacity(0.8))
                    .multilineTextAlignment(.center)
            }
        }
        .padding(18)
        .background(Color.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.cardBorder))
    }
}
