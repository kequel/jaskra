import SwiftUI

// =====================================================================
//  RESULT VIEW — diagnosis summary for a completed analysis
// =====================================================================

struct ResultView: View {
    let result: GlaucomaResult
    let patient: Patient
    var image: UIImage?
    var onNewAnalysis: () -> Void
    var onClose: () -> Void

    @State private var appeared = false

    private var risk: RiskLevel { RiskLevel(cdr: result.cupToDiscRatio) }
    private var accent: Color { result.hasGlaucoma ? .danger : .success }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 22) {
                hero

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous)
                                .strokeBorder(accent.opacity(0.4), lineWidth: 1)
                        )
                        .overlay(alignment: .bottomTrailing) {
                            Chip(text: "Nakładka AI", icon: "sparkles", color: .brand)
                                .padding(12)
                        }
                }

                VStack(spacing: 12) {
                    MetricRow(icon: "chart.pie.fill", label: "Wskaźnik C/D (CDR)",
                              value: String(format: "%.2f", result.cupToDiscRatio), color: risk.color)
                    MetricRow(icon: "waveform.path.ecg", label: "Pewność klasyfikacji",
                              value: "\(Int(result.confidence * 100))%", color: .brand)
                    MetricRow(icon: "exclamationmark.shield.fill", label: "Poziom ryzyka",
                              value: risk.label, color: risk.color)
                }

                CDRScaleView(value: result.cupToDiscRatio)

                assignmentChip
                disclaimer
                buttons
            }
            .padding(.horizontal, DS.screenPadding)
            .padding(.top, 60)
            .padding(.bottom, 36)
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 12)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.4)) { appeared = true }
        }
    }

    // MARK: Sections

    private var hero: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle().fill(accent.opacity(0.12)).frame(width: 92, height: 92)
                Circle().strokeBorder(accent.opacity(0.3), lineWidth: 1).frame(width: 92, height: 92)
                Image(systemName: result.hasGlaucoma ? "exclamationmark.triangle.fill" : "checkmark.seal.fill")
                    .font(.system(size: 38, weight: .light))
                    .foregroundStyle(accent)
            }
            VStack(spacing: 6) {
                Text(result.hasGlaucoma ? "Wykryto cechy jaskry" : "Brak cech jaskry")
                    .font(.system(size: 21, weight: .bold, design: .rounded))
                    .foregroundStyle(accent)
                    .multilineTextAlignment(.center)
                Text("Pewność modelu: \(Int(result.confidence * 100))%")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Color.textSecondary)
            }
        }
    }

    private var assignmentChip: some View {
        HStack(spacing: 10) {
            AvatarView(patient: patient, size: 34)
            VStack(alignment: .leading, spacing: 1) {
                Text("Zapisano w historii pacjenta")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(0.5)
                    .textCase(.uppercase)
                    .foregroundStyle(Color.textTertiary)
                Text(patient.fullName)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
            }
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Color.success)
        }
        .padding(12)
        .glassCard()
    }

    private var disclaimer: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle")
                .foregroundStyle(Color.textTertiary)
                .font(.system(size: 14))
                .padding(.top, 1)
            Text("Wynik jest generowany przez model AI i ma charakter wyłącznie informacyjny. Skonsultuj się z okulistą w celu postawienia diagnozy.")
                .font(.system(size: 12))
                .foregroundStyle(Color.textTertiary)
        }
    }

    private var buttons: some View {
        VStack(spacing: 12) {
            Button {
                onNewAnalysis()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "plus.circle.fill")
                    Text("Nowa analiza")
                }
            }
            .buttonStyle(FilledButtonStyle())

            Button("Zakończ") { onClose() }
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(Color.textSecondary)
        }
    }
}

// MARK: - Metric row

struct MetricRow: View {
    let icon: String
    let label: String
    let value: String
    var color: Color = .brand

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle().fill(color.opacity(0.14)).frame(width: 42, height: 42)
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(color)
            }
            Text(label)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Color.textSecondary)
            Spacer()
            Text(value)
                .font(.system(size: 19, weight: .bold, design: .rounded))
                .foregroundStyle(color)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .glassCard(cornerRadius: DS.radiusSmall + 4)
    }
}

// MARK: - CDR scale

struct CDRScaleView: View {
    let value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("SKALA C/D")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(Color.textTertiary)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.surface2).frame(height: 8)
                    Capsule()
                        .fill(LinearGradient(colors: [.success, .warning, .danger],
                                             startPoint: .leading, endPoint: .trailing))
                        .frame(width: max(8, CGFloat(min(value, 1.0)) * geo.size.width), height: 8)
                    Circle()
                        .fill(.white)
                        .frame(width: 16, height: 16)
                        .shadow(color: .black.opacity(0.3), radius: 4)
                        .offset(x: CGFloat(min(value, 1.0)) * (geo.size.width - 16))
                }
            }
            .frame(height: 16)

            HStack {
                scaleLabel("0.0", "Niskie", .success)
                Spacer()
                scaleLabel("0.5", "Średnie", .warning)
                Spacer()
                scaleLabel("0.8+", "Wysokie", .danger)
            }
        }
        .padding(18)
        .glassCard()
    }

    private func scaleLabel(_ value: String, _ label: String, _ color: Color) -> some View {
        VStack(spacing: 1) {
            Text(value).font(.system(size: 11, weight: .semibold, design: .rounded))
            Text(label).font(.system(size: 10))
        }
        .foregroundStyle(color.opacity(0.85))
    }
}
