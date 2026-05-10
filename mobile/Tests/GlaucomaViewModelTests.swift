import XCTest
import SwiftUI
@testable import AppModule

@MainActor
final class GlaucomaViewModelTests: XCTestCase {
    
    // MARK: - Setup
    var viewModel: GlaucomaViewModel!
    
    override func setUp() {
        super.setUp()
        // Initialize before each test
        viewModel = GlaucomaViewModel()
    }
    
    override func tearDown() {
        // Clean up after each test
        viewModel = nil
        super.tearDown()
    }
    
    // MARK: - Tests
    
    /// Test plan:
    /// - Verify that riskText returns correct string for negative case (CDR < 0.4)
    func test_RiskText_NegativeCase_ReturnsLow() {
        let result = viewModel.riskText(cdr: 0.3)
        XCTAssertEqual(result, "NISKIE", "CDR below 0.4 should return negative (NISKIE) classification")
    }
    
    /// Test plan:
    /// - Verify that riskText returns correct string for moderate case (0.4 <= CDR < 0.6)
    func test_RiskText_ModerateCase_ReturnsModerate() {
        let result = viewModel.riskText(cdr: 0.5)
        XCTAssertEqual(result, "UMIARKOWANE", "CDR between 0.4 and 0.6 should return moderate (UMIARKOWANE) classification")
    }
    
    /// Test plan:
    /// - Verify that riskText returns correct string for positive case (CDR >= 0.6)
    func test_RiskText_PositiveCase_ReturnsHigh() {
        let result = viewModel.riskText(cdr: 0.7)
        XCTAssertEqual(result, "WYSOKIE", "CDR above 0.6 should return positive (WYSOKIE) classification")
    }
    
    /// Test plan:
    /// - Verify riskColor returns positiveGreen for negative case
    func test_RiskColor_NegativeCase_ReturnsPositiveGreen() {
        let result = viewModel.riskColor(cdr: 0.3)
        XCTAssertEqual(result, .positiveGreen, "Low risk should return positiveGreen color")
    }
    
    /// Test plan:
    /// - Verify riskColor returns warningAmber for moderate case
    func test_RiskColor_ModerateCase_ReturnsWarningAmber() {
        let result = viewModel.riskColor(cdr: 0.5)
        XCTAssertEqual(result, .warningAmber, "Moderate risk should return warningAmber color")
    }
    
    /// Test plan:
    /// - Verify riskColor returns dangerRed for positive case
    func test_RiskColor_PositiveCase_ReturnsDangerRed() {
        let result = viewModel.riskColor(cdr: 0.7)
        XCTAssertEqual(result, .dangerRed, "High risk should return dangerRed color")
    }
    
    /// Test plan:
    /// - Verify initial state of ViewModel variables
    func test_InitialState_IsCorrect() {
        XCTAssertNil(viewModel.selectedImage, "Selected image should be nil on init")
        XCTAssertEqual(viewModel.analysisStep, 0, "Analysis step should be 0 on init")
        XCTAssertFalse(viewModel.showPicker, "Show picker should be false on init")
    }
    
    /// Test plan:
    /// - Verify reset function clears state back to defaults
    func test_Reset_ClearsState() {
        // Given
        viewModel.selectedImage = UIImage()
        viewModel.analysisStep = 3
        
        // When
        viewModel.reset()
        
        // Then
        XCTAssertNil(viewModel.selectedImage, "Selected image should be nil after reset")
        XCTAssertEqual(viewModel.analysisStep, 0, "Analysis step should be reset to 0")
    }
}