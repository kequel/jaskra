import XCTest
import SwiftUI
@testable import AppModule

@MainActor
final class GlaucomaViewModelTests: XCTestCase {
    
    func testRiskText() {
        // Setup ViewModel
        let viewModel = GlaucomaViewModel()
        
        // Verify if the function returns correct Polish UI strings for given CDR values
        XCTAssertEqual(viewModel.riskText(cdr: 0.3), "NISKIE")
        XCTAssertEqual(viewModel.riskText(cdr: 0.5), "UMIARKOWANE")
        XCTAssertEqual(viewModel.riskText(cdr: 0.7), "WYSOKIE")
    }
    
    func testRiskColor() {
        // Setup ViewModel
        let viewModel = GlaucomaViewModel()
        
        // Verify if the function assigns correct risk colors based on the CDR threshold
        XCTAssertEqual(viewModel.riskColor(cdr: 0.3), .positiveGreen)
        XCTAssertEqual(viewModel.riskColor(cdr: 0.5), .warningAmber)
        XCTAssertEqual(viewModel.riskColor(cdr: 0.7), .dangerRed)
    }
}