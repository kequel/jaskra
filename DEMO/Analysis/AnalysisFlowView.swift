import SwiftUI
import PhotosUI

// =====================================================================
//  ANALYSIS FLOW — pick image → analyze (stream) → result
//  Tied to a patient (saved to local history immediately), or run ad-hoc
//  with `patient: nil` for a quick, unsaved analysis that can optionally
//  be assigned to a patient afterwards from the result screen.
// =====================================================================

struct AnalysisFlowView: View {
    @EnvironmentObject private var store: PatientStore
    @EnvironmentObject private var auth: AuthManager
    @Environment(\.dismiss) private var dismiss

    /// nil for a quick, ad-hoc analysis that isn't saved until (optionally)
    /// assigned to a patient afterwards.
    let patient: Patient?

    private enum Phase { case picking, analyzing, result, error }

    @State private var phase: Phase = .picking
    @State private var selectedImage: UIImage?
    @State private var processedImage: UIImage?
    @State private var step = 0
    @State private var result: GlaucomaResult?
    @State private var maskImage: UIImage?
    @State private var errorMessage = ""
    /// Patient the (initially unsaved) result got assigned to after the fact.
    @State private var savedPatient: Patient?
    @State private var showAssignSheet = false

    @State private var showCamera = false
    @State private var showPhotosPicker = false
    @State private var photoItem: PhotosPickerItem?
    @State private var analysisTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            ScreenBackground()

            switch phase {
            case .picking:
                pickingView
            case .analyzing:
                AnalyzingView(step: step)
            case .result:
                if let result {
                    ResultView(
                        result: result,
                        patient: patient ?? savedPatient,
                        image: processedImage,
                        onAssign: (patient == nil && savedPatient == nil) ? { showAssignSheet = true } : nil,
                        onNewAnalysis: resetToPicking,
                        onClose: { dismiss() }
                    )
                }
            case .error:
                errorView
            }

            if phase != .result {
                closeButton
            }
        }
        .photosPicker(isPresented: $showPhotosPicker, selection: $photoItem, matching: .images)
        .fullScreenCover(isPresented: $showCamera) {
            ImagePicker(image: $selectedImage, sourceType: .camera).ignoresSafeArea()
        }
        .sheet(isPresented: $showAssignSheet) {
            PatientPickerView { picked in
                if let result {
                    store.addRecord(for: picked, result: result, image: processedImage, maskImage: maskImage, rawImage: selectedImage)
                }
                savedPatient = picked
                showAssignSheet = false
            }
        }
        .onChange(of: photoItem) { _, newItem in
            Task { @MainActor in
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    selectedImage = image
                }
            }
        }
        .onDisappear { analysisTask?.cancel() }
    }

    // MARK: - Picking

    private var pickingView: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 22) {
                header

                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity)
                        .frame(height: 280)
                        .clipShape(RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: DS.radiusCard, style: .continuous)
                                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                        )

                    Button {
                        startAnalysis()
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "waveform.path.ecg.rectangle.fill").font(.system(size: 18))
                            Text("Analizuj")
                        }
                    }
                    .buttonStyle(FilledButtonStyle())

                    Button("Wybierz inne zdjęcie") { selectedImage = nil }
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(Color.textSecondary)
                } else {
                    HStack(spacing: 14) {
                        SourceCard(icon: "camera.fill", title: "Aparat", subtitle: "Zrób zdjęcie", tint: .brand) {
                            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                                showCamera = true
                            } else {
                                showPhotosPicker = true
                            }
                        }
                        SourceCard(icon: "photo.on.rectangle.angled", title: "Galeria", subtitle: "Wybierz zdjęcie", tint: .violet) {
                            showPhotosPicker = true
                        }
                    }
                    .frame(height: 168)

                    Text("Wynik ma charakter informacyjny i nie zastępuje konsultacji lekarskiej.")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.textTertiary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 30)
                }
            }
            .padding(.horizontal, DS.screenPadding)
            .padding(.top, 70)
            .padding(.bottom, 32)
        }
    }

    @ViewBuilder
    private var header: some View {
        if let patient {
            patientHeader(patient)
        } else {
            quickAnalysisHeader
        }
    }

    private func patientHeader(_ patient: Patient) -> some View {
        HStack(spacing: 12) {
            AvatarView(patient: patient, size: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text("Analiza dla")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(1)
                    .textCase(.uppercase)
                    .foregroundStyle(Color.textTertiary)
                Text(patient.fullName)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
            }
            Spacer()
        }
        .padding(12)
        .glassCard()
    }

    private var quickAnalysisHeader: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle().fill(Color.brand.opacity(0.16)).frame(width: 44, height: 44)
                Image(systemName: "bolt.fill")
                    .font(.system(size: 17))
                    .foregroundStyle(Color.brand)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Szybka analiza")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                Text("Wynik możesz przypisać do pacjenta po zakończeniu.")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.textTertiary)
            }
            Spacer()
        }
        .padding(12)
        .glassCard()
    }

    // MARK: - Error

    private var errorView: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle().fill(Color.danger.opacity(0.12)).frame(width: 88, height: 88)
                Image(systemName: "xmark.octagon")
                    .font(.system(size: 36, weight: .ultraLight))
                    .foregroundStyle(Color.danger)
            }
            VStack(spacing: 8) {
                Text("Coś poszło nie tak")
                    .font(.system(size: 19, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.textPrimary)
                Text(errorMessage)
                    .font(.system(size: 13))
                    .foregroundStyle(Color.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 36)
            }
            VStack(spacing: 12) {
                Button {
                    startAnalysis()
                } label: {
                    Label("Spróbuj ponownie", systemImage: "arrow.clockwise")
                }
                .buttonStyle(FilledButtonStyle())

                Button("Wróć do wyboru zdjęcia") { resetToPicking() }
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Color.textSecondary)
            }
            .padding(.horizontal, 36)
        }
    }

    private var closeButton: some View {
        VStack {
            HStack {
                Button { dismiss() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Color.textSecondary)
                        .frame(width: 38, height: 38)
                        .background(.ultraThinMaterial, in: Circle())
                }
                Spacer()
            }
            Spacer()
        }
        .padding(.horizontal, DS.screenPadding)
        .padding(.top, 12)
    }

    // MARK: - Actions

    private func startAnalysis() {
        guard let image = selectedImage else { return }
        step = 0
        errorMessage = ""
        phase = .analyzing

        analysisTask = Task { @MainActor in
            do {
                let res = try await GlaucomaService.shared.analyzeStreaming(
                    image: image,
                    token: auth.token,
                    onStep: { step = $0 }
                )
                let processed = decodeImage(res.imageBase64)
                let mask = res.maskImageBase64.flatMap(decodeImage)
                if let patient {
                    store.addRecord(for: patient, result: res, image: processed, maskImage: mask, rawImage: image)
                }
                self.processedImage = processed
                self.maskImage = mask
                self.result = res
                self.phase = .result
            } catch is CancellationError {
                // View dismissed mid-flight — ignore.
            } catch {
                self.errorMessage = error.localizedDescription
                self.phase = .error
            }
        }
    }

    private func resetToPicking() {
        selectedImage = nil
        processedImage = nil
        maskImage = nil
        result = nil
        savedPatient = nil
        step = 0
        phase = .picking
    }

    private func decodeImage(_ base64: String) -> UIImage? {
        guard let data = Data(base64Encoded: base64) else { return nil }
        return UIImage(data: data)
    }
}

// MARK: - Source card

struct SourceCard: View {
    let icon: String
    let title: String
    let subtitle: String
    var tint: Color = .brand
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                ZStack {
                    Circle().fill(tint.opacity(0.16)).frame(width: 52, height: 52)
                    Image(systemName: icon)
                        .font(.system(size: 21, weight: .light))
                        .foregroundStyle(tint)
                }
                VStack(spacing: 3) {
                    Text(title)
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundStyle(Color.textPrimary)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(Color.textSecondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .glassCard()
        }
        .buttonStyle(.plain)
    }
}
