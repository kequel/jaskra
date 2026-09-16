import Foundation
import UIKit

// =====================================================================
//  DEMO SERVICE — no backend, no network calls.
//  This is a standalone copy of mobile/Services/GlaucomaService.swift for
//  presenting the app while the Azure backend is unreachable. Every method
//  keeps the exact same signature as the real service (so the rest of the
//  app — AuthManager, AnalysisFlowView — needs no changes), but simulates
//  delays and returns plausible fake results instead of calling the API.
// =====================================================================

// MARK: - Response models (kept identical to the real backend contract)

struct GlaucomaResult: Codable {
    let hasGlaucoma: Bool
    let confidence: Double
    let cupToDiscRatio: Double
    let imageBase64: String
    /// Demo-only: a standalone disc/cup mask image (no photo underneath),
    /// used by the analysis comparison screen's "Maski" mode.
    let maskImageBase64: String?

    enum CodingKeys: String, CodingKey {
        case hasGlaucoma     = "has_glaucoma"
        case confidence      = "confidence"
        case cupToDiscRatio  = "cup_to_disc_ratio"
        case imageBase64     = "image_base64"
        case maskImageBase64 = "mask_image_base64"
    }
}

// MARK: - Errors

enum ServiceError: LocalizedError {
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .encodingFailed: return "Nie udało się przetworzyć obrazu."
        }
    }
}

// MARK: - Service

final class GlaucomaService {
    static let shared = GlaucomaService()
    private init() {}

    // MARK: Auth (simulated — any credentials are accepted)

    func register(username: String, password: String) async throws {
        try await Task.sleep(nanoseconds: 500_000_000)
    }

    func login(username: String, password: String) async throws -> String {
        try await Task.sleep(nanoseconds: 500_000_000)
        return "demo-token-\(username)"
    }

    // MARK: History (patients/history are stored locally by PatientStore already)

    func history(token: String) async throws -> [HistoryItem] {
        []
    }

    // MARK: Analysis (simulated progress + a plausible generated result)

    /// Simulates the same step sequence the real backend streams (slowed
    /// down for a live demo — ~3s per step), then returns a result. If the
    /// picked photo is recognized as one of our own reference fundus photos
    /// (see ImageFingerprint.swift), returns the real pipeline result for
    /// it instead of a random mock.
    func analyzeStreaming(
        image: UIImage,
        token: String?,
        onStep: @escaping (Int) -> Void
    ) async throws -> GlaucomaResult {
        let stepDelayNs: UInt64 = 3_000_000_000
        for step in 1...5 {
            await MainActor.run { onStep(step) }
            try await Task.sleep(nanoseconds: stepDelayNs)
        }

        if let matched = ImageFingerprint.matchingSeed(for: image) {
            return GlaucomaResult(
                hasGlaucoma: matched.hasGlaucoma,
                confidence: matched.confidence,
                cupToDiscRatio: matched.cdr,
                imageBase64: matched.imageBase64,
                maskImageBase64: matched.maskBase64
            )
        }

        let cdr = Double.random(in: 0.28...0.82)
        let hasGlaucoma = cdr >= 0.6
        let confidence = Double.random(in: 0.84...0.97)
        let overlay = demoOverlayImage(from: image, cdr: cdr) ?? image
        guard let base64 = overlay.jpegData(compressionQuality: 0.85)?.base64EncodedString() else {
            throw ServiceError.encodingFailed
        }
        let maskBase64 = demoMaskImage(size: image.size, cdr: cdr)?
            .jpegData(compressionQuality: 0.85)?
            .base64EncodedString()

        return GlaucomaResult(
            hasGlaucoma: hasGlaucoma,
            confidence: confidence,
            cupToDiscRatio: cdr,
            imageBase64: base64,
            maskImageBase64: maskBase64
        )
    }

    // MARK: - Helpers

    private func discCupRects(in size: CGSize, cdr: Double) -> (disc: CGRect, cup: CGRect) {
        let side = min(size.width, size.height) * 0.55
        let discRect = CGRect(
            x: (size.width - side) / 2,
            y: (size.height - side) / 2,
            width: side,
            height: side
        )
        let cupSide = side * CGFloat(cdr)
        let cupRect = CGRect(
            x: discRect.midX - cupSide / 2,
            y: discRect.midY - cupSide / 2,
            width: cupSide,
            height: cupSide
        )
        return (discRect, cupRect)
    }

    /// Draws a disc/cup ring pair over the source photo so the demo result
    /// looks like an annotated AI overlay instead of the raw input image.
    private func demoOverlayImage(from image: UIImage, cdr: Double) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { _ in
            image.draw(at: .zero)

            let (discRect, cupRect) = discCupRects(in: image.size, cdr: cdr)
            let lineWidth = min(image.size.width, image.size.height) * 0.55 * 0.02

            let discPath = UIBezierPath(ovalIn: discRect)
            discPath.lineWidth = lineWidth
            UIColor.systemTeal.withAlphaComponent(0.9).setStroke()
            discPath.stroke()

            let cupPath = UIBezierPath(ovalIn: cupRect)
            cupPath.lineWidth = lineWidth
            UIColor.systemYellow.withAlphaComponent(0.9).setStroke()
            cupPath.stroke()
        }
    }

    /// Renders the disc/cup as filled shapes on a plain background (no
    /// photo) — a stand-in segmentation mask for the comparison screen.
    private func demoMaskImage(size: CGSize, cdr: Double) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            UIColor.black.setFill()
            UIBezierPath(rect: CGRect(origin: .zero, size: size)).fill()

            let (discRect, cupRect) = discCupRects(in: size, cdr: cdr)
            UIColor.systemTeal.setFill()
            UIBezierPath(ovalIn: discRect).fill()
            UIColor.systemYellow.setFill()
            UIBezierPath(ovalIn: cupRect).fill()
        }
    }
}
