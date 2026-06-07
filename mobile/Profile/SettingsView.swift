import SwiftUI

// =====================================================================
//  SETTINGS — appearance (light/dark/system), app info, session
// =====================================================================

struct SettingsView: View {
    @EnvironmentObject private var theme: ThemeManager
    @EnvironmentObject private var auth: AuthManager

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 24) {

                    // Appearance
                    VStack(alignment: .leading, spacing: 12) {
                        SectionHeader(title: "Wygląd")
                        HStack(spacing: 12) {
                            ForEach(AppearanceMode.allCases) { mode in
                                AppearanceOption(mode: mode, isSelected: theme.mode == mode) {
                                    withAnimation(.easeInOut(duration: 0.25)) { theme.mode = mode }
                                }
                            }
                        }
                        Text("Efekt szkła (glass) działa w obu trybach.")
                            .font(.system(size: 12))
                            .foregroundStyle(Color.textTertiary)
                    }

                    // App info
                    VStack(alignment: .leading, spacing: 12) {
                        SectionHeader(title: "Informacje")
                        VStack(spacing: 0) {
                            InfoRow(label: "Wersja aplikacji", value: "2.0")
                            Divider().overlay(Color.border)
                            InfoRow(label: "Model AI", value: "YOLO + UNet++")
                            Divider().overlay(Color.border)
                            InfoRow(label: "Tryb", value: auth.isLoggedIn ? "Lekarz" : "Gość")
                        }
                        .glassCard()
                    }

                    // Disclaimer
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "shield.lefthalf.filled")
                            .foregroundStyle(Color.textTertiary)
                            .font(.system(size: 14))
                            .padding(.top, 1)
                        Text("Aplikacja ma charakter pomocniczy i nie zastępuje diagnozy lekarskiej. Zdjęcia są wysyłane do analizy z usuniętymi metadanymi (EXIF/GPS).")
                            .font(.system(size: 12))
                            .foregroundStyle(Color.textTertiary)
                    }

                    if auth.isLoggedIn {
                        Button {
                            auth.logout()
                        } label: {
                            Label("Wyloguj się", systemImage: "rectangle.portrait.and.arrow.right")
                        }
                        .buttonStyle(SoftButtonStyle(tint: .danger))
                    }
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.top, 4)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Ustawienia")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Components

struct AppearanceOption: View {
    let mode: AppearanceMode
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 10) {
                Image(systemName: mode.icon)
                    .font(.system(size: 22, weight: .regular))
                    .foregroundStyle(isSelected ? Color.brand : Color.textSecondary)
                Text(mode.label)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(isSelected ? Color.textPrimary : Color.textSecondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous)
                    .fill(isSelected ? Color.brand.opacity(0.14) : Color.surface.opacity(0.5))
            )
            .overlay(
                RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous)
                    .strokeBorder(isSelected ? Color.brand : Color.border, lineWidth: isSelected ? 1.5 : 1)
            )
        }
        .buttonStyle(.plain)
    }
}

struct InfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 15, design: .rounded))
                .foregroundStyle(Color.textPrimary)
            Spacer()
            Text(value)
                .font(.system(size: 15, weight: .medium, design: .rounded))
                .foregroundStyle(Color.textSecondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}
