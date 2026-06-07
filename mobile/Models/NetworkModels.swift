import Foundation

// =====================================================================
//  Auxiliary networking models (auth + history responses).
//  Kept out of GlaucomaService.swift so the CI "swift-backend-sync" job
//  (which greps GlaucomaService.swift for CodingKeys) only sees the
//  GlaucomaResult contract.
// =====================================================================

struct HistoryItem: Decodable, Identifiable {
    let id: Int
    let isGlaucoma: Bool
    let cdr: Double
    let date: String?

    enum CodingKeys: String, CodingKey {
        case id
        case isGlaucoma = "is_glaucoma"
        case cdr
        case date
    }
}

struct HistoryResponse: Decodable {
    let history: [HistoryItem]
}

struct TokenResponse: Decodable {
    let accessToken: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
    }
}
