import SwiftUI
import UIKit

@MainActor
class GlaucomaViewModel: ObservableObject {
    @Published var screen: AppScreen = .home
    @Published var selectedImage: UIImage?
    @Published var showPicker = false

    func runAnalysis() {
        guard let image = selectedImage else { return }
        screen = .analyzing
        Task {
            do {
                let result = try await GlaucomaService.shared.analyze(image: image)
                screen = .result(result)
            } catch {
                screen = .error(error.localizedDescription)
            }
        }
    }

    func reset() {
        selectedImage = nil
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
