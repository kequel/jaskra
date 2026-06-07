import SwiftUI

// =====================================================================
//  PATIENTS LIST — home tab. Browse patients, start a new analysis,
//  add a new patient.
// =====================================================================

struct PatientsListView: View {
    @EnvironmentObject private var store: PatientStore

    private enum ActiveSheet: Identifiable {
        case newPatient, picker
        var id: Int { hashValue }
    }

    @State private var search = ""
    @State private var activeSheet: ActiveSheet?
    @State private var analysisPatient: Patient?

    private var filtered: [Patient] {
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !query.isEmpty else { return store.patients }
        return store.patients.filter {
            $0.fullName.lowercased().contains(query) || $0.email.lowercased().contains(query)
        }
    }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 16) {
                    Button {
                        activeSheet = .picker
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "viewfinder.circle.fill").font(.system(size: 20))
                            Text("Nowa analiza")
                        }
                    }
                    .buttonStyle(FilledButtonStyle())
                    .padding(.top, 4)

                    if store.patients.isEmpty {
                        EmptyStateView(
                            icon: "person.2",
                            title: "Brak pacjentów",
                            message: "Dodaj pierwszego pacjenta, aby rozpocząć analizy i prowadzić historię badań.",
                            tint: .brand
                        )
                    } else {
                        SectionHeader(title: "Pacjenci (\(store.patients.count))")

                        ForEach(filtered) { patient in
                            NavigationLink(value: patient) {
                                PatientRow(patient: patient, count: store.analysisCount(for: patient.id))
                            }
                            .buttonStyle(.plain)
                        }

                        if filtered.isEmpty {
                            Text("Brak wyników dla „\(search)”.")
                                .font(.system(size: 13))
                                .foregroundStyle(Color.textTertiary)
                                .padding(.top, 8)
                        }
                    }
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Pacjenci")
        .searchable(text: $search, prompt: "Szukaj pacjenta")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    activeSheet = .newPatient
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 16, weight: .semibold))
                }
            }
        }
        .navigationDestination(for: Patient.self) { patient in
            PatientDetailView(patient: patient)
        }
        .sheet(item: $activeSheet) { sheet in
            switch sheet {
            case .newPatient:
                PatientFormView { _ in }
            case .picker:
                PatientPickerView { picked in
                    // Close the picker, then present the analysis flow.
                    // The short delay lets the sheet finish dismissing before
                    // the full-screen cover appears (avoids a presentation race).
                    activeSheet = nil
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        analysisPatient = picked
                    }
                }
            }
        }
        .fullScreenCover(item: $analysisPatient) { patient in
            AnalysisFlowView(patient: patient)
        }
    }
}

// MARK: - Patient row

struct PatientRow: View {
    let patient: Patient
    let count: Int

    var body: some View {
        HStack(spacing: 14) {
            AvatarView(patient: patient, size: 52)

            VStack(alignment: .leading, spacing: 3) {
                Text(patient.fullName)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                Text(patient.email.isEmpty ? "—" : patient.email)
                    .font(.system(size: 12))
                    .foregroundStyle(Color.textSecondary)
                    .lineLimit(1)
            }

            Spacer()

            VStack(spacing: 1) {
                Text("\(count)")
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.brand)
                Text(count == 1 ? "analiza" : "analiz")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.textTertiary)
            }

            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.textTertiary)
        }
        .padding(14)
        .glassCard()
    }
}

// =====================================================================
//  PATIENT PICKER — choose an existing patient or create a new one,
//  used when starting a new analysis from the home tab.
//  Calls `onPick` with the chosen patient; the parent dismisses it.
// =====================================================================

struct PatientPickerView: View {
    @EnvironmentObject private var store: PatientStore
    @Environment(\.dismiss) private var dismiss

    let onPick: (Patient) -> Void

    @State private var search = ""
    @State private var showNewPatient = false

    private var filtered: [Patient] {
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !query.isEmpty else { return store.patients }
        return store.patients.filter {
            $0.fullName.lowercased().contains(query) || $0.email.lowercased().contains(query)
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                ScreenBackground()

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 14) {
                        Button {
                            showNewPatient = true
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: "person.fill.badge.plus").font(.system(size: 18))
                                Text("Nowy pacjent")
                            }
                        }
                        .buttonStyle(SoftButtonStyle(tint: .violet))
                        .padding(.top, 4)

                        if store.patients.isEmpty {
                            EmptyStateView(
                                icon: "person.crop.circle.badge.plus",
                                title: "Brak pacjentów",
                                message: "Utwórz pacjenta, aby przypisać do niego analizę.",
                                tint: .violet
                            )
                        } else {
                            ForEach(filtered) { patient in
                                Button {
                                    onPick(patient)
                                } label: {
                                    PatientRow(patient: patient, count: store.analysisCount(for: patient.id))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(.horizontal, DS.screenPadding)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle("Wybierz pacjenta")
            .navigationBarTitleDisplayMode(.inline)
            .searchable(text: $search, prompt: "Szukaj pacjenta")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Anuluj") { dismiss() }
                }
            }
            .sheet(isPresented: $showNewPatient) {
                PatientFormView { created in
                    onPick(created)
                }
            }
        }
    }
}
