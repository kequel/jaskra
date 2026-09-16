import SwiftUI

// =====================================================================
//  HISTORY — every analysis across all patients, newest first
// =====================================================================

struct HistoryView: View {
    @EnvironmentObject private var store: PatientStore

    private var records: [AnalysisRecord] { store.allRecordsByDate }

    var body: some View {
        ZStack {
            ScreenBackground()

            if records.isEmpty {
                EmptyStateView(
                    icon: "clock.arrow.circlepath",
                    title: "Brak historii",
                    message: "Wykonane analizy pojawią się tutaj — wraz z przypisanym pacjentem.",
                    tint: .indigo
                )
            } else {
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 14) {
                        SectionHeader(title: "Wszystkie analizy (\(records.count))")

                        ForEach(records) { record in
                            if let patient = store.patient(by: record.patientId) {
                                NavigationLink(value: record) {
                                    HistoryRow(record: record, patient: patient, image: store.image(for: record))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(.horizontal, DS.screenPadding)
                    .padding(.top, 4)
                    .padding(.bottom, 32)
                }
            }
        }
        .navigationTitle("Historia")
        .navigationDestination(for: AnalysisRecord.self) { record in
            RecordDetailView(record: record, patient: store.patient(by: record.patientId))
        }
    }
}

// MARK: - History row (includes the patient)

struct HistoryRow: View {
    let record: AnalysisRecord
    let patient: Patient
    var image: UIImage?

    var body: some View {
        HStack(spacing: 14) {
            AvatarView(patient: patient, size: 48)

            VStack(alignment: .leading, spacing: 5) {
                Text(patient.fullName)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(AppFormat.dateTime.string(from: record.date))
                        .font(.system(size: 11))
                        .foregroundStyle(Color.textTertiary)
                }
                DiagnosisBadge(hasGlaucoma: record.hasGlaucoma)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.2f", record.cupToDiscRatio))
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .foregroundStyle(record.risk.color)
                Text("CDR")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(Color.textTertiary)
            }
        }
        .padding(12)
        .glassCard()
    }
}
