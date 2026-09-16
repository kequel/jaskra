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

    private var filtered: [Patient] { store.patients.matching(search) }

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

                        AlphabetPatientList(patients: filtered) { patient in
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
        .searchable(text: $search, prompt: "Szukaj po imieniu lub nazwisku")
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

// MARK: - Search & alphabetical grouping

extension Array where Element == Patient {
    /// Matches patients whose first- or last-name TOKEN *starts with* the
    /// query — searching "ga" finds "Gabriel", not "Agata". Email is never
    /// searched, only the name.
    func matching(_ query: String) -> [Patient] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return self }
        return filter { patient in
            let tokens = "\(patient.firstName) \(patient.lastName)"
                .lowercased()
                .split(separator: " ")
            return tokens.contains { $0.hasPrefix(q) }
        }
    }
}

private struct AlphabetGroup: Identifiable {
    let letter: String
    let patients: [Patient]
    var id: String { letter }
}

private func alphabetGroups(of patients: [Patient]) -> [AlphabetGroup] {
    let groups = Dictionary(grouping: patients) { patient -> String in
        let letter = patient.lastName.trimmingCharacters(in: .whitespaces).first
            ?? patient.firstName.trimmingCharacters(in: .whitespaces).first
        return letter.map { String($0).uppercased() } ?? "#"
    }
    return groups.keys.sorted().map { key in
        let sorted = groups[key]!.sorted { lhs, rhs in
            let byLastName = lhs.lastName.localizedCaseInsensitiveCompare(rhs.lastName)
            if byLastName != .orderedSame { return byLastName == .orderedAscending }
            return lhs.firstName.localizedCaseInsensitiveCompare(rhs.firstName) == .orderedAscending
        }
        return AlphabetGroup(letter: key, patients: sorted)
    }
}

/// Contacts-style patient list: sectioned A–Z by surname, each letter
/// collapsible independently.
struct AlphabetPatientList<RowContent: View>: View {
    let patients: [Patient]
    @ViewBuilder var rowContent: (Patient) -> RowContent

    @State private var collapsedLetters: Set<String> = []

    private var groups: [AlphabetGroup] { alphabetGroups(of: patients) }

    var body: some View {
        VStack(spacing: 18) {
            ForEach(groups) { group in
                VStack(spacing: 10) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) { toggle(group.letter) }
                    } label: {
                        HStack(spacing: 10) {
                            Text(group.letter)
                                .font(.system(size: 13, weight: .bold, design: .rounded))
                                .foregroundStyle(Color.brand)
                                .frame(width: 24, height: 24)
                                .background(Color.brand.opacity(0.14), in: Circle())
                            Text("\(group.patients.count)")
                                .font(.system(size: 12))
                                .foregroundStyle(Color.textTertiary)
                            Spacer()
                            Image(systemName: isCollapsed(group.letter) ? "chevron.right" : "chevron.down")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(Color.textTertiary)
                        }
                    }
                    .buttonStyle(.plain)

                    if !isCollapsed(group.letter) {
                        VStack(spacing: 10) {
                            ForEach(group.patients) { rowContent($0) }
                        }
                    }
                }
            }
        }
    }

    private func isCollapsed(_ letter: String) -> Bool { collapsedLetters.contains(letter) }

    private func toggle(_ letter: String) {
        if collapsedLetters.contains(letter) { collapsedLetters.remove(letter) }
        else { collapsedLetters.insert(letter) }
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

    private var filtered: [Patient] { store.patients.matching(search) }

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
                            AlphabetPatientList(patients: filtered) { patient in
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
            .searchable(text: $search, prompt: "Szukaj po imieniu lub nazwisku")
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
