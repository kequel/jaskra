import SwiftUI

// =====================================================================
//  DOMAIN MODELS — Patient, avatars, analysis records, risk levels
//  Patients are stored locally on the device (the backend does not yet
//  model patients — see docs/backend-issue-patients.md).
// =====================================================================

// MARK: - Avatar kind (SF Symbols figure family)

enum AvatarKind: String, Codable, CaseIterable, Identifiable {
    case woman, man, girl, boy

    var id: String { rawValue }

    var label: String {
        switch self {
        case .woman: return "Kobieta"
        case .man:   return "Mężczyzna"
        case .girl:  return "Dziewczynka"
        case .boy:   return "Chłopiec"
        }
    }

    /// SF Symbol used to represent this person.
    var symbol: String {
        switch self {
        case .woman: return "figure.stand.dress"
        case .man:   return "figure.stand"
        case .girl:  return "figure.child"
        case .boy:   return "figure.child"
        }
    }

    /// A sensible default tint so freshly created avatars look varied.
    var defaultTint: AvatarTint {
        switch self {
        case .woman: return .rose
        case .man:   return .sky
        case .girl:  return .coral
        case .boy:   return .mint
        }
    }
}

// MARK: - Avatar tint (maps onto the multi-color accent palette)

enum AvatarTint: String, Codable, CaseIterable, Identifiable {
    case teal, coral, amber, violet, indigo, rose, mint, sky

    var id: String { rawValue }

    var color: Color {
        switch self {
        case .teal:   return .brand
        case .coral:  return .coral
        case .amber:  return .amber
        case .violet: return .violet
        case .indigo: return .indigo
        case .rose:   return .rose
        case .mint:   return .mint
        case .sky:    return .sky
        }
    }
}

// MARK: - Patient

struct Patient: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var firstName: String
    var lastName: String
    var email: String
    var avatarKind: AvatarKind
    var avatarTint: AvatarTint
    var createdAt: Date = Date()

    var fullName: String {
        let name = "\(firstName) \(lastName)".trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? "Pacjent bez nazwy" : name
    }

    var initials: String {
        let f = firstName.first.map(String.init) ?? ""
        let l = lastName.first.map(String.init) ?? ""
        let value = (f + l).uppercased()
        return value.isEmpty ? "?" : value
    }
}

// MARK: - Analysis record (local history, one per completed analysis)

struct AnalysisRecord: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var patientId: UUID
    var date: Date = Date()
    var hasGlaucoma: Bool
    var confidence: Double
    var cupToDiscRatio: Double
    /// File name of the processed (overlay) image saved in the app's storage.
    var imageFilename: String?

    var risk: RiskLevel { RiskLevel(cdr: cupToDiscRatio) }
}

// MARK: - Risk level (single source of truth for CDR thresholds)

enum RiskLevel: String {
    case low, moderate, high

    init(cdr: Double) {
        switch cdr {
        case ..<0.4:  self = .low
        case ..<0.6:  self = .moderate
        default:      self = .high
        }
    }

    var label: String {
        switch self {
        case .low:      return "NISKIE"
        case .moderate: return "UMIARKOWANE"
        case .high:     return "WYSOKIE"
        }
    }

    var color: Color {
        switch self {
        case .low:      return .success
        case .moderate: return .warning
        case .high:     return .danger
        }
    }
}
