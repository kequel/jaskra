import XCTest
import SwiftUI
@testable import AppModule

@MainActor
final class TestGlaucomaViewModel: XCTestCase {
    /*
    Unit tests for the GlaucomaViewModel logic.
    Verifies risk text classifications, color mapping, and state reset behavior based on CDR thresholds.
    */

    var viewModel: GlaucomaViewModel!

    override func setUp() {
        super.setUp()
        viewModel = GlaucomaViewModel()
    }

    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }

    func test_risk_text_negative_returns_low() {
        /*
        TEST NAME: test_risk_text_negative_returns_low
        COMPONENT: GlaucomaViewModel.riskText(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskText() with CDR values 0.3 and 0.0.
        3. Assert the returned text equals "NISKIE".
        */
        XCTAssertEqual(viewModel.riskText(cdr: 0.3), "NISKIE")
        XCTAssertEqual(viewModel.riskText(cdr: 0.0), "NISKIE")
    }

    func test_risk_text_boundary_low_returns_moderate() {
        /*
        TEST NAME: test_risk_text_boundary_low_returns_moderate
        COMPONENT: GlaucomaViewModel.riskText(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskText() with exactly 0.4.
        3. Assert the returned text equals "UMIARKOWANE".
        */
        XCTAssertEqual(viewModel.riskText(cdr: 0.4), "UMIARKOWANE")
    }

    func test_risk_text_moderate_returns_moderate() {
        /*
        TEST NAME: test_risk_text_moderate_returns_moderate
        COMPONENT: GlaucomaViewModel.riskText(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskText() with CDR value 0.5.
        3. Assert the returned text equals "UMIARKOWANE".
        */
        XCTAssertEqual(viewModel.riskText(cdr: 0.5), "UMIARKOWANE")
    }

    func test_risk_text_boundary_high_returns_high() {
        /*
        TEST NAME: test_risk_text_boundary_high_returns_high
        COMPONENT: GlaucomaViewModel.riskText(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskText() with exactly 0.6.
        3. Assert the returned text equals "WYSOKIE".
        */
        XCTAssertEqual(viewModel.riskText(cdr: 0.6), "WYSOKIE")
    }

    func test_risk_text_positive_returns_high() {
        /*
        TEST NAME: test_risk_text_positive_returns_high
        COMPONENT: GlaucomaViewModel.riskText(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskText() with CDR values 0.7 and 1.0.
        3. Assert the returned text equals "WYSOKIE".
        */
        XCTAssertEqual(viewModel.riskText(cdr: 0.7), "WYSOKIE")
        XCTAssertEqual(viewModel.riskText(cdr: 1.0), "WYSOKIE")
    }

    func test_risk_color_negative_returns_green() {
        /*
        TEST NAME: test_risk_color_negative_returns_green
        COMPONENT: GlaucomaViewModel.riskColor(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskColor() with CDR values 0.3 and 0.0.
        3. Assert the returned color equals positiveGreen.
        */
        XCTAssertEqual(viewModel.riskColor(cdr: 0.3), .positiveGreen)
        XCTAssertEqual(viewModel.riskColor(cdr: 0.0), .positiveGreen)
    }

    func test_risk_color_boundary_low_returns_amber() {
        /*
        TEST NAME: test_risk_color_boundary_low_returns_amber
        COMPONENT: GlaucomaViewModel.riskColor(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskColor() with exactly 0.4.
        3. Assert the returned color equals warningAmber.
        */
        XCTAssertEqual(viewModel.riskColor(cdr: 0.4), .warningAmber)
    }

    func test_risk_color_moderate_returns_amber() {
        /*
        TEST NAME: test_risk_color_moderate_returns_amber
        COMPONENT: GlaucomaViewModel.riskColor(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskColor() with CDR value 0.5.
        3. Assert the returned color equals warningAmber.
        */
        XCTAssertEqual(viewModel.riskColor(cdr: 0.5), .warningAmber)
    }

    func test_risk_color_boundary_high_returns_red() {
        /*
        TEST NAME: test_risk_color_boundary_high_returns_red
        COMPONENT: GlaucomaViewModel.riskColor(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskColor() with exactly 0.6.
        3. Assert the returned color equals dangerRed.
        */
        XCTAssertEqual(viewModel.riskColor(cdr: 0.6), .dangerRed)
    }

    func test_risk_color_positive_returns_red() {
        /*
        TEST NAME: test_risk_color_positive_returns_red
        COMPONENT: GlaucomaViewModel.riskColor(cdr:)
        1. Initialize GlaucomaViewModel.
        2. Call riskColor() with CDR values 0.7 and 1.0.
        3. Assert the returned color equals dangerRed.
        */
        XCTAssertEqual(viewModel.riskColor(cdr: 0.7), .dangerRed)
        XCTAssertEqual(viewModel.riskColor(cdr: 1.0), .dangerRed)
    }

    func test_initial_state_returns_defaults() {
        /*
        TEST NAME: test_initial_state_returns_defaults
        COMPONENT: GlaucomaViewModel.init()
        1. Initialize GlaucomaViewModel.
        2. Assert selectedImage is nil.
        3. Assert analysisStep equals 0.
        4. Assert showPicker is false.
        */
        XCTAssertNil(viewModel.selectedImage)
        XCTAssertEqual(viewModel.analysisStep, 0)
        XCTAssertFalse(viewModel.showPicker)
    }

    func test_reset_clears_state() {
        /*
        TEST NAME: test_reset_clears_state
        COMPONENT: GlaucomaViewModel.reset()
        1. Initialize GlaucomaViewModel.
        2. Assign mock values to selectedImage and analysisStep.
        3. Call reset().
        4. Assert selectedImage is nil and analysisStep equals 0.
        */
        viewModel.selectedImage = UIImage()
        viewModel.analysisStep = 3

        viewModel.reset()

        XCTAssertNil(viewModel.selectedImage)
        XCTAssertEqual(viewModel.analysisStep, 0)
    }
}
