import SwiftUI

// =====================================================================
//  PATIENT FORM — create or edit a patient (name, surname, email, avatar)
// =====================================================================

struct PatientFormView: View {
    @EnvironmentObject private var store: PatientStore
    @Environment(\.dismiss) private var dismiss

    /// Patient being edited, or nil when creating a new one.
    var editing: Patient?
    /// Called with the saved patient (useful when creating during an analysis flow).
    var onSaved: (Patient) -> Void

    @State private var firstName: String
    @State private var lastName: String
    @State private var email: String
    @State private var avatarKind: AvatarKind
    @State private var avatarTint: AvatarTint

    init(editing: Patient? = nil, onSaved: @escaping (Patient) -> Void) {
        self.editing = editing
        self.onSaved = onSaved
        _firstName = State(initialValue: editing?.firstName ?? "")
        _lastName  = State(initialValue: editing?.lastName ?? "")
        _email     = State(initialValue: editing?.email ?? "")
        _avatarKind = State(initialValue: editing?.avatarKind ?? .woman)
        _avatarTint = State(initialValue: editing?.avatarTint ?? AvatarKind.woman.defaultTint)
    }

    private var canSave: Bool {
        !firstName.trimmingCharacters(in: .whitespaces).isEmpty
            || !lastName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var body: some View {
        NavigationStack {
            ZStack {
                ScreenBackground()

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 22) {
                        // Live avatar preview
                        VStack(spacing: 10) {
                            AvatarView(kind: avatarKind, tint: avatarTint, size: 92)
                            Text(previewName)
                                .font(.system(size: 16, weight: .semibold, design: .rounded))
                                .foregroundStyle(Color.textPrimary)
                        }
                        .padding(.top, 8)

                        VStack(spacing: 14) {
                            GlassField(title: "Imię", icon: "person.fill", text: $firstName)
                            GlassField(title: "Nazwisko", icon: "person.text.rectangle.fill", text: $lastName)
                            GlassField(
                                title: "E-mail",
                                icon: "envelope.fill",
                                keyboard: .emailAddress,
                                textContentType: .emailAddress,
                                text: $email
                            )
                        }

                        // Avatar kind
                        VStack(alignment: .leading, spacing: 10) {
                            SectionHeader(title: "Awatar")
                            HStack(spacing: 12) {
                                ForEach(AvatarKind.allCases) { kind in
                                    Button {
                                        avatarKind = kind
                                        avatarTint = kind.defaultTint
                                    } label: {
                                        VStack(spacing: 6) {
                                            AvatarView(kind: kind, tint: avatarKind == kind ? avatarTint : kind.defaultTint, size: 54)
                                                .overlay(
                                                    Circle().strokeBorder(Color.brand, lineWidth: avatarKind == kind ? 2 : 0)
                                                )
                                            Text(kind.label)
                                                .font(.system(size: 10, weight: .medium))
                                                .foregroundStyle(avatarKind == kind ? Color.textPrimary : Color.textTertiary)
                                        }
                                        .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }

                        // Avatar tint
                        VStack(alignment: .leading, spacing: 10) {
                            SectionHeader(title: "Kolor")
                            HStack(spacing: 12) {
                                ForEach(AvatarTint.allCases) { tint in
                                    Button {
                                        avatarTint = tint
                                    } label: {
                                        Circle()
                                            .fill(tint.color)
                                            .frame(width: 30, height: 30)
                                            .overlay(
                                                Circle().strokeBorder(Color.textPrimary.opacity(avatarTint == tint ? 0.9 : 0), lineWidth: 2)
                                            )
                                            .overlay(
                                                Image(systemName: "checkmark")
                                                    .font(.system(size: 12, weight: .bold))
                                                    .foregroundStyle(.white)
                                                    .opacity(avatarTint == tint ? 1 : 0)
                                            )
                                    }
                                    .buttonStyle(.plain)
                                    .frame(maxWidth: .infinity)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, DS.screenPadding)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle(editing == nil ? "Nowy pacjent" : "Edytuj pacjenta")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Anuluj") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Zapisz", action: save)
                        .font(.system(size: 16, weight: .semibold))
                        .disabled(!canSave)
                }
            }
        }
    }

    private var previewName: String {
        let name = "\(firstName) \(lastName)".trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? "Nowy pacjent" : name
    }

    private func save() {
        var patient = editing ?? Patient(
            firstName: "",
            lastName: "",
            email: "",
            avatarKind: avatarKind,
            avatarTint: avatarTint
        )
        patient.firstName = firstName.trimmingCharacters(in: .whitespaces)
        patient.lastName = lastName.trimmingCharacters(in: .whitespaces)
        patient.email = email.trimmingCharacters(in: .whitespaces)
        patient.avatarKind = avatarKind
        patient.avatarTint = avatarTint

        if editing == nil {
            store.addPatient(patient)
        } else {
            store.updatePatient(patient)
        }
        onSaved(patient)
        dismiss()
    }
}
