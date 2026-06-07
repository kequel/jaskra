import SwiftUI

// =====================================================================
//  RECORD DETAIL — a single past analysis from the history
// =====================================================================

struct RecordDetailView: View {
    @EnvironmentObject private var store: PatientStore
    @Environment(\.dismiss) private var dismiss

    let record: AnalysisRecord
    var patient: Patient?

    @State private var showDeleteConfirm = false

    private var accent: Color { record.hasGlaucoma ? .danger : .success }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 20) {
                    if let patient {
                        HStack(spacing: 12) {
                            AvatarView(patient: patient, size: 44)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(patient.fullName)
                                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                                    .foregroundStyle(Color.textPrimary)
                                Text(AppFormat.dateTime.string(from: record.date))
                                    .font(.system(size: 12))
                                    .foregroundStyle(Color.textSecondary)
                            }
                            Spacer()
                        }
                        .padding(12)
                        .glassCard()
                    }

                    HStack(spacing: 12) {
                        DiagnosisBadge(hasGlaucoma: record.hasGlaucoma)
                        Chip(text: "Ryzyko: \(record.risk.label.capitalized)", color: record.risk.color)
                        Spacer()
                    }

                    if let image = store.image(for: record) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous)
                                    .strokeBorder(accent.opacity(0.4), lineWidth: 1)
                            )
                    }

                    VStack(spacing: 12) {
                        MetricRow(icon: "chart.pie.fill", label: "Wskaźnik C/D (CDR)",
                                  value: String(format: "%.2f", record.cupToDiscRatio), color: record.risk.color)
                        MetricRow(icon: "waveform.path.ecg", label: "Pewność klasyfikacji",
                                  value: "\(Int(record.confidence * 100))%", color: .brand)
                        MetricRow(icon: "exclamationmark.shield.fill", label: "Poziom ryzyka",
                                  value: record.risk.label, color: record.risk.color)
                    }

                    CDRScaleView(value: record.cupToDiscRatio)
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.top, 4)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Wynik analizy")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(role: .destructive) {
                    showDeleteConfirm = true
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 15, weight: .semibold))
                }
            }
        }
        .confirmationDialog("Usunąć tę analizę?", isPresented: $showDeleteConfirm, titleVisibility: .visible) {
            Button("Usuń", role: .destructive) {
                store.deleteRecord(record)
                dismiss()
            }
            Button("Anuluj", role: .cancel) {}
        }
    }
}
