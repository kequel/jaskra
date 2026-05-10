import Foundation
import UIKit

// Response Model
struct GlaucomaResult: Codable {
    let hasGlaucoma: Bool
    let confidence: Double
    let cupToDiscRatio: Double
    let imageBase64: String

    enum CodingKeys: String, CodingKey {
        case hasGlaucoma    = "has_glaucoma"
        case confidence     = "confidence"
        case cupToDiscRatio = "cup_to_disc_ratio"
        case imageBase64    = "image_base64"
    }
}

// Streaming progress model
struct StreamProgress: Decodable {
    let status: String
    let step: Int?
    let message: String?
    let data: GlaucomaResult?
}

// App State
enum AppScreen {
    case home
    case analyzing
    case result(GlaucomaResult)
    case error(String)
}

// API Service
class GlaucomaService {
    static let shared = GlaucomaService()
    private init() {}

    private let baseURL = "https://glaucoma-a5cpf7arbdetdmax.polandcentral-01.azurewebsites.net"

    // Streaming analyze (NDJSON)
    func analyzeStreaming(
        image: UIImage,
        onStep: @escaping (Int) -> Void
    ) async throws -> GlaucomaResult {
        guard let url = URL(string: "\(baseURL)/analyze-glaucoma-stream") else { throw ServiceError.badURL }
        guard let jpeg = image.jpegData(compressionQuality: 0.9) else { throw ServiceError.encodingFailed }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"eye.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpeg)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw ServiceError.serverError
        }

        var finalResult: GlaucomaResult?

        for try await line in asyncBytes.lines {
            guard !line.isEmpty,
                  let data = line.data(using: .utf8) else { continue }

            guard let progress = try? JSONDecoder().decode(StreamProgress.self, from: data) else { continue }

            if let step = progress.step {
                await MainActor.run { onStep(step) }
            }

            if progress.status == "success", let result = progress.data {
                finalResult = result
            } else if progress.status == "error" {
                throw ServiceError.serverError
            }
        }

        guard let result = finalResult else { throw ServiceError.serverError }
        return result
    }

    // Classic (non-streaming) fallback
    func analyze(image: UIImage) async throws -> GlaucomaResult {
        guard let url = URL(string: "\(baseURL)/analyze-glaucoma") else { throw ServiceError.badURL }
        guard let jpeg = image.jpegData(compressionQuality: 0.9) else { throw ServiceError.encodingFailed }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"eye.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpeg)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw ServiceError.serverError
        }

        return try JSONDecoder().decode(GlaucomaResult.self, from: data)
    }
}

enum ServiceError: LocalizedError {
    case badURL, encodingFailed, serverError
    var errorDescription: String? {
        switch self {
        case .badURL:           return "Nieprawidłowy adres serwera."
        case .encodingFailed:   return "Nie udało się przetworzyć obrazu."
        case .serverError:      return "Serwer zwrócił błąd. Spróbuj ponownie."
        }
    }
}