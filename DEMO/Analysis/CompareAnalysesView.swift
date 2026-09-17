import SwiftUI

// =====================================================================
//  COMPARE ANALYSES (demo only) — pick two analyses for a patient and
//  drag a slider to reveal how the eye (or its disc/cup mask) changed
//  between them, alongside the CDR/diagnosis delta.
// =====================================================================

struct CompareAnalysesView: View {
    @EnvironmentObject private var store: PatientStore

    let patient: Patient
    /// This patient's records, newest first (as returned by PatientStore).
    let records: [AnalysisRecord]

    private enum Mode: String, CaseIterable, Identifiable {
        case photos = "Zdjęcia"
        case masks = "Maski"
        var id: String { rawValue }
    }

    @State private var mode: Mode = .photos
    @State private var olderIndex: Int
    @State private var newerIndex: Int = 0

    init(patient: Patient, records: [AnalysisRecord]) {
        self.patient = patient
        self.records = records
        _olderIndex = State(initialValue: max(records.count - 1, 0))
    }

    private var older: AnalysisRecord? { records.indices.contains(olderIndex) ? records[olderIndex] : nil }
    private var newer: AnalysisRecord? { records.indices.contains(newerIndex) ? records[newerIndex] : nil }

    private func displayImage(for record: AnalysisRecord) -> UIImage? {
        switch mode {
        case .photos: return store.rawImage(for: record)
        case .masks:  return store.maskImage(for: record)
        }
    }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 20) {
                    Picker("Tryb porównania", selection: $mode) {
                        ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    recordPickers

                    if let older, let newer,
                       let beforeImage = displayImage(for: older),
                       let afterImage = displayImage(for: newer) {
                        BeforeAfterSlider(beforeImage: beforeImage, afterImage: afterImage)
                        metricsComparison(older: older, newer: newer)
                    } else {
                        EmptyStateView(
                            icon: "photo.on.rectangle",
                            title: "Brak obrazu",
                            message: "Dla wybranej analizy nie zapisano zdjęcia.",
                            tint: .violet
                        )
                    }
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.top, 12)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Porównanie analiz")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Record pickers

    private var recordPickers: some View {
        VStack(spacing: 10) {
            recordPickerRow(title: "Archiwalna", selection: $olderIndex)
            recordPickerRow(title: "Aktualna", selection: $newerIndex)
        }
    }

    private func recordPickerRow(title: String, selection: Binding<Int>) -> some View {
        HStack {
            Text(title)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(Color.textSecondary)
            Spacer()
            Picker(title, selection: selection) {
                ForEach(records.indices, id: \.self) { idx in
                    Text(AppFormat.dateTime.string(from: records[idx].date)).tag(idx)
                }
            }
            .pickerStyle(.menu)
            .tint(Color.brand)
        }
        .padding(12)
        .glassCard()
    }

    // MARK: - Metrics

    private func metricsComparison(older: AnalysisRecord, newer: AnalysisRecord) -> some View {
        let delta = newer.cupToDiscRatio - older.cupToDiscRatio

        return VStack(spacing: 14) {
            HStack(spacing: 14) {
                metricColumn(label: "CDR wcześniej", value: String(format: "%.2f", older.cupToDiscRatio), color: older.risk.color)
                Image(systemName: "arrow.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.textTertiary)
                metricColumn(label: "CDR teraz", value: String(format: "%.2f", newer.cupToDiscRatio), color: newer.risk.color)
            }

            HStack(spacing: 8) {
                Image(systemName: delta > 0.005 ? "arrow.up.right" : delta < -0.005 ? "arrow.down.right" : "minus")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(delta > 0.005 ? Color.danger : delta < -0.005 ? Color.success : Color.textTertiary)
                Text("\(String(format: "%+.2f", delta)) CDR od poprzedniej analizy")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
            }

            HStack(spacing: 10) {
                DiagnosisBadge(hasGlaucoma: older.hasGlaucoma)
                Image(systemName: "arrow.right")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.textTertiary)
                DiagnosisBadge(hasGlaucoma: newer.hasGlaucoma)
            }
        }
        .padding(16)
        .glassCard()
    }

    private func metricColumn(label: String, value: String, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(Color.textTertiary)
        }
        .frame(maxWidth: .infinity)
    }
}

// =====================================================================
//  BEFORE/AFTER SLIDER — classic drag-to-reveal image comparison.
// =====================================================================

struct BeforeAfterSlider: View {
    let beforeImage: UIImage
    let afterImage: UIImage

    @State private var position: CGFloat = 0.5

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            let handleX = width * position

            ZStack(alignment: .topLeading) {
                Image(uiImage: afterImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: height)
                    .clipped()

                Image(uiImage: beforeImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: height)
                    .clipped()
                    .mask(alignment: .leading) {
                        Rectangle().frame(width: handleX)
                    }

                Rectangle()
                    .fill(Color.white.opacity(0.9))
                    .frame(width: 2)
                    .offset(x: handleX - 1)

                labels

                Circle()
                    .fill(Color.white)
                    .frame(width: 34, height: 34)
                    .shadow(color: .black.opacity(0.3), radius: 6, x: 0, y: 2)
                    .overlay(
                        Image(systemName: "arrow.left.and.right")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(Color.black.opacity(0.7))
                    )
                    .offset(x: handleX - 17, y: height / 2 - 17)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { value in
                    position = min(max(0, value.location.x / width), 1)
                }
            )
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private var labels: some View {
        VStack {
            HStack {
                Chip(text: "ARCHIWALNA", icon: "clock.arrow.circlepath", color: .violet)
                Spacer()
                Chip(text: "AKTUALNA", icon: "sparkles", color: .brand)
            }
            Spacer()
        }
        .padding(10)
    }
}
