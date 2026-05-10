import XCTest
import SwiftUI
@testable import AppModule

@MainActor
final class GlaucomaViewModelTests: XCTestCase {
    
    // MARK: - Setup
    var viewModel: GlaucomaViewModel!
    
    override func setUp() {
        super.setUp()
        viewModel = GlaucomaViewModel()
    }
    
    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }
    
    // MARK: - Risk Text Tests
    
    func test_RiskText_NegativeCase_ReturnsLow() {
        XCTAssertEqual(viewModel.riskText(cdr: 0.3), "NISKIE", "CDR below 0.4 should return negative (NISKIE)")
        XCTAssertEqual(viewModel.riskText(cdr: 0.0), "NISKIE", "CDR at edge 0.0 should return negative (NISKIE)")
    }
    
    func test_RiskText_BoundaryLow_ReturnsModerate() {
        XCTAssertEqual(viewModel.riskText(cdr: 0.4), "UMIARKOWANE", "CDR at exactly 0.4 should return moderate (UMIARKOWANE)")
    }
    
    func test_RiskText_ModerateCase_ReturnsModerate() {
        XCTAssertEqual(viewModel.riskText(cdr: 0.5), "UMIARKOWANE", "CDR between 0.4 and 0.6 should return moderate (UMIARKOWANE)")
    }
    
    func test_RiskText_BoundaryHigh_ReturnsHigh() {
        XCTAssertEqual(viewModel.riskText(cdr: 0.6), "WYSOKIE", "CDR at exactly 0.6 should return positive (WYSOKIE)")
    }
    
    func test_RiskText_PositiveCase_ReturnsHigh() {
        XCTAssertEqual(viewModel.riskText(cdr: 0.7), "WYSOKIE", "CDR above 0.6 should return positive (WYSOKIE)")
        XCTAssertEqual(viewModel.riskText(cdr: 1.0), "WYSOKIE", "CDR at edge 1.0 should return positive (WYSOKIE)")
    }
    
    // MARK: - Risk Color Tests
    
    func test_RiskColor_NegativeCase_ReturnsPositiveGreen() {
        XCTAssertEqual(viewModel.riskColor(cdr: 0.3), .positiveGreen, "Low risk should return positiveGreen color")
        XCTAssertEqual(viewModel.riskColor(cdr: 0.0), .positiveGreen, "Edge 0.0 should return positiveGreen color")
    }
    
    func test_RiskColor_BoundaryLow_ReturnsWarningAmber() {
        XCTAssertEqual(viewModel.riskColor(cdr: 0.4), .warningAmber, "CDR exactly 0.4 should return warningAmber color")
    }
    
    func test_RiskColor_ModerateCase_ReturnsWarningAmber() {
        XCTAssertEqual(viewModel.riskColor(cdr: 0.5), .warningAmber, "Moderate risk should return warningAmber color")
    }
    
    func test_RiskColor_BoundaryHigh_ReturnsDangerRed() {
        XCTAssertEqual(viewModel.riskColor(cdr: 0.6), .dangerRed, "CDR exactly 0.6 should return dangerRed color")
    }
    
    func test_RiskColor_PositiveCase_ReturnsDangerRed() {
        XCTAssertEqual(viewModel.riskColor(cdr: 0.7), .dangerRed, "High risk should return dangerRed color")
        XCTAssertEqual(viewModel.riskColor(cdr: 1.0), .dangerRed, "Edge 1.0 should return dangerRed color")
    }
    
    // MARK: - State & Reset Tests
    
    func test_InitialState_IsCorrect() {
        XCTAssertNil(viewModel.selectedImage, "Selected image should be nil on init")
        XCTAssertEqual(viewModel.analysisStep, 0, "Analysis step should be 0 on init")
        XCTAssertFalse(viewModel.showPicker, "Show picker should be false on init")
    }
    
    func test_Reset_ClearsState() {
        // Given
        viewModel.selectedImage = UIImage()
        viewModel.analysisStep = 3
        
        // When
        viewModel.reset()
        
        // Then - We only assert properties that are explicitly handled by reset() in the current implementation
        XCTAssertNil(viewModel.selectedImage, "Selected image should be nil after reset")
        XCTAssertEqual(viewModel.analysisStep, 0, "Analysis step should be reset to 0")
    }
}