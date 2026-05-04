import SwiftUI
import UIKit

@MainActor
class GlaucomaViewModel: ObservableObject {
    @Published var screen: AppScreen = .home
    @Published var selectedImage: UIImage?
    @Published var showPicker = false

    /// Current backend step (1–5), drives AnalyzingView progress
    @Published var analysisStep: Int = 0

    func runAnalysis() {
        guard let image = selectedImage else { return }
        analysisStep = 0
        screen = .analyzing
        Task {
            do {
                let result = try await GlaucomaService.shared.analyzeStreaming(
                    image: image,
                    onStep: { [weak self] step in
                        self?.analysisStep = step
                    }
                )
                screen = .result(result)
            } catch {
                screen = .error(error.localizedDescription)
            }
        }
    }

    func reset() {
        selectedImage = nil
        analysisStep = 0
        screen = .home
    }

    func resultUIImage(from base64: String) -> UIImage? {
        guard let data = Data(base64Encoded: base64) else { return nil }
        return UIImage(data: data)
    }

    func riskText(cdr: Double) -> String {
        cdr < 0.4 ? "NISKIE" : cdr < 0.6 ? "UMIARKOWANE" : "WYSOKIE"
    }

    func riskColor(cdr: Double) -> Color {
        cdr < 0.4 ? .positiveGreen : cdr < 0.6 ? .warningAmber : .dangerRed
    }
}