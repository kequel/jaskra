import SwiftUI

// =====================================================================
//  AUTH SCREEN — login / register, with "analyze as guest" at the bottom
// =====================================================================

struct AuthView: View {
    @EnvironmentObject private var auth: AuthManager

    enum Mode: String, CaseIterable, Identifiable {
        case login, register
        var id: String { rawValue }
        var title: String { self == .login ? "Logowanie" : "Rejestracja" }
        var cta: String { self == .login ? "Zaloguj się" : "Utwórz konto" }
    }

    @State private var mode: Mode = .login
    @State private var username = ""
    @State private var password = ""
    @State private var isLoading = false
    @State private var errorMessage: String?

    private var canSubmit: Bool {
        !username.trimmingCharacters(in: .whitespaces).isEmpty
            && password.count >= 4
            && !isLoading
    }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 28) {
                    header

                    VStack(spacing: 18) {
                        Picker("", selection: $mode) {
                            ForEach(Mode.allCases) { Text($0.title).tag($0) }
                        }
                        .pickerStyle(.segmented)

                        VStack(spacing: 14) {
                            GlassField(
                                title: "Nazwa użytkownika",
                                icon: "person.fill",
                                textContentType: .username,
                                text: $username
                            )
                            GlassField(
                                title: "Hasło",
                                icon: "lock.fill",
                                isSecure: true,
                                textContentType: mode == .login ? .password : .newPassword,
                                text: $password
                            )
                        }

                        if let errorMessage {
                            Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(Color.danger)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        Button(action: submit) {
                            HStack(spacing: 8) {
                                if isLoading { ProgressView().tint(.white) }
                                Text(mode.cta)
                            }
                        }
                        .buttonStyle(FilledButtonStyle(isEnabled: canSubmit))
                        .disabled(!canSubmit)
                    }
                    .padding(20)
                    .glassCard()
                    .padding(.horizontal, DS.screenPadding)

                    guestSection
                }
                .padding(.top, 60)
                .padding(.bottom, 40)
            }
        }
    }

    // MARK: Sections

    private var header: some View {
        VStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Color.brand.opacity(0.16))
                    .frame(width: 76, height: 76)
                Image(systemName: "eye.fill")
                    .font(.system(size: 32, weight: .light))
                    .foregroundStyle(Color.brand)
            }
            VStack(spacing: 4) {
                Text("Jaskra")
                    .font(.system(size: 26, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                Text("Panel lekarza")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .tracking(1.5)
                    .textCase(.uppercase)
                    .foregroundStyle(Color.textSecondary)
            }
        }
    }

    private var guestSection: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Rectangle().fill(Color.border).frame(height: 1)
                Text("lub").font(.system(size: 12)).foregroundStyle(Color.textTertiary)
                Rectangle().fill(Color.border).frame(height: 1)
            }
            .padding(.horizontal, DS.screenPadding)

            Button {
                auth.continueAsGuest()
            } label: {
                Text("Analizuj jako gość")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
            }
            .buttonStyle(SoftButtonStyle(tint: .violet))
            .padding(.horizontal, DS.screenPadding)

            Text("Tryb gościa nie zapisuje wyników na koncie lekarza.")
                .font(.system(size: 11))
                .foregroundStyle(Color.textTertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }

    // MARK: Actions

    private func submit() {
        errorMessage = nil
        isLoading = true
        let name = username.trimmingCharacters(in: .whitespaces)
        Task { @MainActor in
            do {
                switch mode {
                case .login:    try await auth.login(username: name, password: password)
                case .register: try await auth.register(username: name, password: password)
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
}
