import SwiftUI

// =====================================================================
//  PROFILE — doctor identity, quick stats, settings & session
// =====================================================================

struct ProfileView: View {
    @EnvironmentObject private var auth: AuthManager
    @EnvironmentObject private var store: PatientStore

    @State private var showLogoutConfirm = false

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 18) {
                    doctorCard

                    HStack(spacing: 14) {
                        StatCard(value: "\(store.patients.count)", label: "Pacjenci", icon: "person.2.fill", color: .brand)
                        StatCard(value: "\(store.records.count)", label: "Analizy", icon: "waveform.path.ecg", color: .violet)
                    }

                    VStack(spacing: 12) {
                        NavigationLink {
                            SettingsView()
                        } label: {
                            SettingsRow(icon: "gearshape.fill", title: "Ustawienia", tint: .sky)
                        }
                        .buttonStyle(.plain)
                    }

                    sessionButton
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.top, 4)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Profil")
        .confirmationDialog("Wylogować się?", isPresented: $showLogoutConfirm, titleVisibility: .visible) {
            Button("Wyloguj", role: .destructive) { auth.logout() }
            Button("Anuluj", role: .cancel) {}
        }
    }

    // MARK: Sections

    private var doctorCard: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle().fill(Color.brand.opacity(0.16)).frame(width: 88, height: 88)
                Image(systemName: auth.isLoggedIn ? "stethoscope" : "person.fill.questionmark")
                    .font(.system(size: 36, weight: .light))
                    .foregroundStyle(Color.brand)
            }
            VStack(spacing: 4) {
                Text(auth.isLoggedIn ? (auth.username ?? "Lekarz") : "Tryb gościa")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                Text(auth.isLoggedIn ? "Zalogowany lekarz" : "Wyniki nie są zapisywane na koncie")
                    .font(.system(size: 13))
                    .foregroundStyle(Color.textSecondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .glassCard()
        .padding(.top, 4)
    }

    @ViewBuilder
    private var sessionButton: some View {
        if auth.isLoggedIn {
            Button {
                showLogoutConfirm = true
            } label: {
                Label("Wyloguj się", systemImage: "rectangle.portrait.and.arrow.right")
            }
            .buttonStyle(SoftButtonStyle(tint: .danger))
        } else {
            Button {
                auth.exitGuest()
            } label: {
                Label("Zaloguj się", systemImage: "person.crop.circle.badge.checkmark")
            }
            .buttonStyle(FilledButtonStyle())
        }
    }
}

// MARK: - Components

struct StatCard: View {
    let value: String
    let label: String
    let icon: String
    var color: Color = .brand

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Circle().fill(color.opacity(0.16)).frame(width: 40, height: 40)
                Image(systemName: icon).font(.system(size: 17)).foregroundStyle(color)
            }
            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(Color.textPrimary)
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Color.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
        .glassCard()
    }
}

struct SettingsRow: View {
    let icon: String
    let title: String
    var tint: Color = .brand

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(tint.opacity(0.16))
                    .frame(width: 38, height: 38)
                Image(systemName: icon).font(.system(size: 16)).foregroundStyle(tint)
            }
            Text(title)
                .font(.system(size: 16, weight: .medium, design: .rounded))
                .foregroundStyle(Color.textPrimary)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.textTertiary)
        }
        .padding(14)
        .glassCard()
    }
}
