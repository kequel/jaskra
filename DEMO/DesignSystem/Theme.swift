import SwiftUI
import UIKit

// =====================================================================
//  THEME / PALETTE
//  Two coherent palettes (light + dark) stored in one place.
//  Change a hex value here and it propagates across the whole app.
//  Each semantic color carries a light + dark variant and is resolved
//  automatically at render time via a dynamic UIColor.
// =====================================================================

/// A single semantic color with a light and a dark hex value (0xRRGGBB).
struct ThemeColor {
    let light: UInt32
    let dark: UInt32
}

/// The app palette. Edit values here to retheme the entire app.
enum Palette {

    // MARK: Surfaces
    static let background = ThemeColor(light: 0xF3F4F8, dark: 0x0A0B0F)
    static let surface    = ThemeColor(light: 0xFFFFFF, dark: 0x15161D)
    static let surface2   = ThemeColor(light: 0xEAECF2, dark: 0x1E2029)
    static let border     = ThemeColor(light: 0xE0E3EC, dark: 0x2A2C37)

    // MARK: Text
    static let textPrimary   = ThemeColor(light: 0x16181F, dark: 0xF2F4F8)
    static let textSecondary = ThemeColor(light: 0x596073, dark: 0x9CA1B0)
    static let textTertiary  = ThemeColor(light: 0x969CAC, dark: 0x5C6172)

    // MARK: Brand + multi-color accents
    static let brand  = ThemeColor(light: 0x0D9488, dark: 0x2DD4BF) // teal
    static let coral  = ThemeColor(light: 0xF15B43, dark: 0xFF7A63)
    static let amber  = ThemeColor(light: 0xE08A00, dark: 0xFBBF24)
    static let violet = ThemeColor(light: 0x7C3AED, dark: 0xA78BFA)
    static let indigo = ThemeColor(light: 0x4F46E5, dark: 0x818CF8)
    static let rose   = ThemeColor(light: 0xE11D6E, dark: 0xFB6FA0)
    static let mint   = ThemeColor(light: 0x059669, dark: 0x34D399)
    static let sky    = ThemeColor(light: 0x0284C7, dark: 0x38BDF8)

    // MARK: Semantic
    static let success = ThemeColor(light: 0x059669, dark: 0x34D399)
    static let warning = ThemeColor(light: 0xD97706, dark: 0xFBBF24)
    static let danger  = ThemeColor(light: 0xDC2626, dark: 0xF87171)
}

// =====================================================================
//  Color / UIColor helpers
// =====================================================================

extension UIColor {
    /// Create a solid UIColor from a 0xRRGGBB integer.
    convenience init(rgb: UInt32) {
        let r = CGFloat((rgb >> 16) & 0xFF) / 255.0
        let g = CGFloat((rgb >> 8) & 0xFF) / 255.0
        let b = CGFloat(rgb & 0xFF) / 255.0
        self.init(red: r, green: g, blue: b, alpha: 1.0)
    }
}

extension Color {
    /// Build an adaptive Color that switches between the light/dark hex
    /// automatically based on the active interface style.
    init(_ token: ThemeColor) {
        self = Color(UIColor { traits in
            UIColor(rgb: traits.userInterfaceStyle == .dark ? token.dark : token.light)
        })
    }

    // Semantic accessors — use these everywhere in the UI.
    static let bg            = Color(Palette.background)
    static let surface       = Color(Palette.surface)
    static let surface2      = Color(Palette.surface2)
    static let border        = Color(Palette.border)

    static let textPrimary   = Color(Palette.textPrimary)
    static let textSecondary = Color(Palette.textSecondary)
    static let textTertiary  = Color(Palette.textTertiary)

    static let brand   = Color(Palette.brand)
    static let coral   = Color(Palette.coral)
    static let amber   = Color(Palette.amber)
    static let violet  = Color(Palette.violet)
    static let indigo  = Color(Palette.indigo)
    static let rose    = Color(Palette.rose)
    static let mint    = Color(Palette.mint)
    static let sky     = Color(Palette.sky)

    static let success = Color(Palette.success)
    static let warning = Color(Palette.warning)
    static let danger  = Color(Palette.danger)
}

// =====================================================================
//  Appearance mode (System / Light / Dark) — toggled in Settings
// =====================================================================

enum AppearanceMode: String, CaseIterable, Identifiable {
    case system, light, dark

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "Systemowy"
        case .light:  return "Jasny"
        case .dark:   return "Ciemny"
        }
    }

    var icon: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light:  return "sun.max.fill"
        case .dark:   return "moon.stars.fill"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }
}

/// Holds the user's appearance preference and persists it.
final class ThemeManager: ObservableObject {
    private let storageKey = "appearanceMode"

    @Published var mode: AppearanceMode {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: storageKey) }
    }

    init() {
        let saved = UserDefaults.standard.string(forKey: storageKey)
        mode = AppearanceMode(rawValue: saved ?? "") ?? .system
    }

    var colorScheme: ColorScheme? { mode.colorScheme }
}

// =====================================================================
//  Spacing / radius design tokens
// =====================================================================

enum DS {
    static let radiusCard: CGFloat = 22
    static let radiusButton: CGFloat = 16
    static let radiusSmall: CGFloat = 12
    static let screenPadding: CGFloat = 20
}
