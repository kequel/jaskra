import SwiftUI

// =====================================================================
//  REUSABLE UI COMPONENTS
//  Glass cards, buttons, chips, headers, ambient background.
//  All adapt to light/dark automatically through the Theme palette.
// =====================================================================

// MARK: - Glass card

struct GlassCardModifier: ViewModifier {
    var cornerRadius: CGFloat = DS.radiusCard
    var strokeOpacity: Double = 0.12

    func body(content: Content) -> some View {
        content
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(strokeOpacity), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.16), radius: 16, x: 0, y: 8)
    }
}

extension View {
    /// Apple-style frosted glass card.
    func glassCard(cornerRadius: CGFloat = DS.radiusCard, strokeOpacity: Double = 0.12) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius, strokeOpacity: strokeOpacity))
    }
}

// MARK: - Ambient multi-color background

/// Base background with soft, multi-color ambient glows. Works in both modes.
struct ScreenBackground: View {
    var body: some View {
        ZStack {
            Color.bg.ignoresSafeArea()

            GeometryReader { geo in
                ZStack {
                    glow(.brand, 0.18, size: 340)
                        .offset(x: -130, y: -170)
                    glow(.violet, 0.16, size: 300)
                        .offset(x: geo.size.width - 120, y: -70)
                    glow(.coral, 0.12, size: 280)
                        .offset(x: geo.size.width - 60, y: geo.size.height - 200)
                    glow(.sky, 0.10, size: 260)
                        .offset(x: -90, y: geo.size.height - 140)
                }
            }
            .ignoresSafeArea()
        }
    }

    private func glow(_ color: Color, _ opacity: Double, size: CGFloat) -> some View {
        Circle()
            .fill(color.opacity(opacity))
            .frame(width: size, height: size)
            .blur(radius: 120)
    }
}

// MARK: - Buttons

struct FilledButtonStyle: ButtonStyle {
    var tint: Color = .brand
    var isEnabled: Bool = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 16, weight: .semibold, design: .rounded))
            .foregroundStyle(isEnabled ? Color.white : Color.textTertiary)
            .frame(maxWidth: .infinity)
            .frame(height: 54)
            .background(
                isEnabled ? AnyShapeStyle(tint.gradient) : AnyShapeStyle(Color.surface2),
                in: RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous)
            )
            .shadow(color: isEnabled ? tint.opacity(0.35) : .clear, radius: 14, x: 0, y: 6)
            .opacity(configuration.isPressed ? 0.9 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

struct SoftButtonStyle: ButtonStyle {
    var tint: Color = .brand

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .semibold, design: .rounded))
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous))
            .opacity(configuration.isPressed ? 0.85 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

// MARK: - Chips & pills

struct Chip: View {
    let text: String
    var icon: String? = nil
    var color: Color = .brand

    var body: some View {
        HStack(spacing: 5) {
            if let icon { Image(systemName: icon).font(.system(size: 11, weight: .semibold)) }
            Text(text).font(.system(size: 12, weight: .semibold, design: .rounded))
        }
        .foregroundStyle(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(color.opacity(0.15), in: Capsule())
    }
}

/// Colored diagnosis badge used on result/history rows.
struct DiagnosisBadge: View {
    let hasGlaucoma: Bool

    var body: some View {
        Chip(
            text: hasGlaucoma ? "Cechy jaskry" : "Bez cech",
            icon: hasGlaucoma ? "exclamationmark.triangle.fill" : "checkmark.seal.fill",
            color: hasGlaucoma ? .danger : .success
        )
    }
}

// MARK: - Section header

struct SectionHeader: View {
    let title: String
    var actionTitle: String? = nil
    var action: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .tracking(1)
                .foregroundStyle(Color.textSecondary)
                .textCase(.uppercase)
            Spacer()
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.brand)
            }
        }
    }
}

// MARK: - Labeled text field (glass)

struct GlassField: View {
    let title: String
    var icon: String? = nil
    var keyboard: UIKeyboardType = .default
    var isSecure: Bool = false
    var textContentType: UITextContentType? = nil
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .tracking(1)
                .foregroundStyle(Color.textTertiary)

            HStack(spacing: 10) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 15))
                        .foregroundStyle(Color.textSecondary)
                        .frame(width: 20)
                }
                Group {
                    if isSecure {
                        SecureField("", text: $text)
                    } else {
                        TextField("", text: $text)
                    }
                }
                .font(.system(size: 16, design: .rounded))
                .foregroundStyle(Color.textPrimary)
                .keyboardType(keyboard)
                .textContentType(textContentType)
                .autocorrectionDisabled()
                .textInputAutocapitalization(keyboard == .emailAddress ? .never : .sentences)
            }
            .padding(.horizontal, 14)
            .frame(height: 52)
            .background(Color.surface.opacity(0.7), in: RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: DS.radiusButton, style: .continuous)
                    .strokeBorder(Color.border, lineWidth: 1)
            )
        }
    }
}

// MARK: - Empty state

struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    var tint: Color = .brand

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle().fill(tint.opacity(0.14)).frame(width: 74, height: 74)
                Image(systemName: icon)
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(tint)
            }
            Text(title)
                .font(.system(size: 17, weight: .bold, design: .rounded))
                .foregroundStyle(Color.textPrimary)
            Text(message)
                .font(.system(size: 13))
                .foregroundStyle(Color.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }
}
