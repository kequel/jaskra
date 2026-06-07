import SwiftUI

// =====================================================================
//  PATIENT DETAIL — header, "new analysis", and this patient's history
// =====================================================================

struct PatientDetailView: View {
    @EnvironmentObject private var store: PatientStore
    @Environment(\.dismiss) private var dismiss

    let patient: Patient

    @State private var showAnalysis = false
    @State private var showEdit = false
    @State private var showDeleteConfirm = false

    /// Always read the freshest copy so edits are reflected immediately.
    private var current: Patient { store.patient(by: patient.id) ?? patient }
    private var records: [AnalysisRecord] { store.records(for: patient.id) }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 18) {
                    headerCard

                    Button {
                        showAnalysis = true
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "viewfinder.circle.fill").font(.system(size: 20))
                            Text("Nowa analiza")
                        }
                    }
                    .buttonStyle(FilledButtonStyle())

                    SectionHeader(title: "Historia badań (\(records.count))")

                    if records.isEmpty {
                        EmptyStateView(
                            icon: "waveform.path.ecg",
                            title: "Brak analiz",
                            message: "Wykonaj pierwszą analizę dna oka dla tego pacjenta.",
                            tint: .brand
                        )
                    } else {
                        ForEach(records) { record in
                            NavigationLink(value: record) {
                                RecordRow(record: record, image: store.image(for: record))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle(current.fullName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        showEdit = true
                    } label: {
                        Label("Edytuj pacjenta", systemImage: "pencil")
                    }
                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        Label("Usuń pacjenta", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 16, weight: .semibold))
                }
            }
        }
        .navigationDestination(for: AnalysisRecord.self) { record in
            RecordDetailView(record: record, patient: current)
        }
        .fullScreenCover(isPresented: $showAnalysis) {
            AnalysisFlowView(patient: current)
        }
        .sheet(isPresented: $showEdit) {
            PatientFormView(editing: current) { _ in }
        }
        .confirmationDialog(
            "Usunąć pacjenta i całą jego historię?",
            isPresented: $showDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Usuń", role: .destructive) {
                store.deletePatient(current)
                dismiss()
            }
            Button("Anuluj", role: .cancel) {}
        }
    }

    // MARK: Header

    private var headerCard: some View {
        VStack(spacing: 14) {
            AvatarView(patient: current, size: 88)

            VStack(spacing: 4) {
                Text(current.fullName)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                if !current.email.isEmpty {
                    Text(current.email)
                        .font(.system(size: 13))
                        .foregroundStyle(Color.textSecondary)
                }
            }

            HStack(spacing: 10) {
                Chip(text: current.avatarKind.label, icon: "person.fill", color: current.avatarTint.color)
                Chip(text: "\(records.count) analiz", icon: "waveform.path.ecg", color: .brand)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .glassCard()
        .padding(.top, 4)
    }
}

// MARK: - Record row

struct RecordRow: View {
    let record: AnalysisRecord
    var image: UIImage?

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(record.risk.color.opacity(0.14))
                    .frame(width: 52, height: 52)
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 52, height: 52)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                } else {
                    Image(systemName: "eye.fill")
                        .font(.system(size: 20, weight: .light))
                        .foregroundStyle(record.risk.color)
                }
            }

            VStack(alignment: .leading, spacing: 5) {
                Text(AppFormat.dateTime.string(from: record.date))
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
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

// MARK: - Shared date formatter

enum AppFormat {
    static let dateTime: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "pl_PL")
        f.dateFormat = "d MMM yyyy, HH:mm"
        return f
    }()
}
