import Foundation
import UIKit
import ImageIO

// =====================================================================
//  NETWORKING — talks to the FastAPI backend
//  Endpoints: /register, /login, /history, /analyze-glaucoma-stream
// =====================================================================

// MARK: - Response models

// NOTE: keep these four CodingKeys in sync with the backend response.
// The CI job "swift-backend-sync" greps this file for them.
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

struct StreamProgress: Decodable {
    let status: String
    let step: Int?
    let message: String?
    let data: GlaucomaResult?
}

// NOTE: HistoryItem / TokenResponse live in NetworkModels.swift so this file
// only contains the GlaucomaResult contract that the CI "swift-backend-sync"
// job verifies against the backend.

// MARK: - Errors

enum ServiceError: LocalizedError {
    case badURL
    case encodingFailed
    case unauthorized
    case server(String)

    var errorDescription: String? {
        switch self {
        case .badURL:          return "Nieprawidłowy adres serwera."
        case .encodingFailed:  return "Nie udało się przetworzyć obrazu."
        case .unauthorized:    return "Nieprawidłowa nazwa użytkownika lub hasło."
        case .server(let m):   return m
        }
    }
}

// MARK: - Service

final class GlaucomaService {
    static let shared = GlaucomaService()
    private init() {}

    private let baseURL = "https://glaucoma-a5cpf7arbdetdmax.polandcentral-01.azurewebsites.net"

    // MARK: Auth

    /// Register a new account. Throws ServiceError.server with backend detail on failure.
    func register(username: String, password: String) async throws {
        guard let url = URL(string: "\(baseURL)/register") else { throw ServiceError.badURL }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = formBody(["username": username, "password": password])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ServiceError.server("Brak odpowiedzi serwera.") }

        if !(200...299).contains(http.statusCode) {
            throw ServiceError.server(detail(from: data) ?? "Rejestracja nie powiodła się (\(http.statusCode)).")
        }
    }

    /// Log in and return a JWT access token.
    func login(username: String, password: String) async throws -> String {
        guard let url = URL(string: "\(baseURL)/login") else { throw ServiceError.badURL }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = formBody(["username": username, "password": password])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ServiceError.server("Brak odpowiedzi serwera.") }

        if http.statusCode == 401 { throw ServiceError.unauthorized }
        guard (200...299).contains(http.statusCode) else {
            throw ServiceError.server(detail(from: data) ?? "Logowanie nie powiodło się (\(http.statusCode)).")
        }

        let token = try JSONDecoder().decode(TokenResponse.self, from: data)
        return token.accessToken
    }

    // MARK: History

    func history(token: String) async throws -> [HistoryItem] {
        guard let url = URL(string: "\(baseURL)/history") else { throw ServiceError.badURL }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ServiceError.server("Brak odpowiedzi serwera.") }
        if http.statusCode == 401 { throw ServiceError.unauthorized }
        guard (200...299).contains(http.statusCode) else {
            throw ServiceError.server("Nie udało się pobrać historii (\(http.statusCode)).")
        }

        return try JSONDecoder().decode(HistoryResponse.self, from: data).history
    }

    // MARK: Analysis (streaming NDJSON)

    /// Run the streaming analysis. Pass a token to have the backend persist
    /// the result to the doctor's history; pass nil for an anonymous run.
    func analyzeStreaming(
        image: UIImage,
        token: String?,
        onStep: @escaping (Int) -> Void
    ) async throws -> GlaucomaResult {
        guard let url = URL(string: "\(baseURL)/analyze-glaucoma-stream") else { throw ServiceError.badURL }
        guard let jpeg = strippedJPEG(from: image) else { throw ServiceError.encodingFailed }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

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
            throw ServiceError.server("Serwer zwrócił błąd. Spróbuj ponownie.")
        }

        var finalResult: GlaucomaResult?

        for try await line in asyncBytes.lines {
            guard !line.isEmpty, let data = line.data(using: .utf8) else { continue }
            guard let progress = try? JSONDecoder().decode(StreamProgress.self, from: data) else { continue }

            if let step = progress.step {
                await MainActor.run { onStep(step) }
            }
            if progress.status == "success", let result = progress.data {
                finalResult = result
            } else if progress.status == "error" {
                throw ServiceError.server(progress.message ?? "Wystąpił błąd serwera.")
            }
        }

        guard let result = finalResult else { throw ServiceError.server("Serwer nie zwrócił wyniku.") }
        return result
    }

    // MARK: - Helpers

    /// Re-encode a UIImage to JPEG with EXIF/GPS/TIFF/IPTC metadata removed.
    private func strippedJPEG(from image: UIImage, quality: CGFloat = 0.9) -> Data? {
        guard let jpeg = image.jpegData(compressionQuality: quality),
              let source = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let uti = CGImageSourceGetType(source) else { return nil }

        let output = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(output, uti, 1, nil) else { return nil }

        let cleanProperties: [CFString: Any] = [
            kCGImagePropertyExifDictionary: kCFNull,
            kCGImagePropertyGPSDictionary: kCFNull,
            kCGImagePropertyTIFFDictionary: kCFNull,
            kCGImagePropertyIPTCDictionary: kCFNull
        ]
        CGImageDestinationAddImageFromSource(dest, source, 0, cleanProperties as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return output as Data
    }

    private func formBody(_ params: [String: String]) -> Data {
        let allowed = CharacterSet(charactersIn:
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        func enc(_ s: String) -> String { s.addingPercentEncoding(withAllowedCharacters: allowed) ?? s }
        let joined = params.map { "\(enc($0.key))=\(enc($0.value))" }.joined(separator: "&")
        return Data(joined.utf8)
    }

    /// Extracts FastAPI's {"detail": "..."} message if present.
    private func detail(from data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let s = obj["detail"] as? String { return s }
        if let s = obj["message"] as? String { return s }
        return nil
    }
}
