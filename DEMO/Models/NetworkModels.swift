import Foundation

// =====================================================================
//  Auxiliary model kept for API parity with the real GlaucomaService
//  (history() returns [HistoryItem] there too). Patients/history are
//  actually stored locally by PatientStore in this demo build.
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
