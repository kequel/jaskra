import SwiftUI
import PhotosUI

// =====================================================================
//  ANALYSIS FLOW — pick image → analyze (stream) → result
//  Always tied to a specific patient. Saves the result to local history
//  and (when the doctor is logged in) to the backend.
// =====================================================================

struct AnalysisFlowView: View {
    @EnvironmentObject private var store: PatientStore
    @EnvironmentObject private var auth: AuthManager
    @Environment(\.dismiss) private var dismiss

    let patient: Patient

    private enum Phase { case picking, analyzing, result, error }

    @State private var phase: Phase = .picking
    @State private var selectedImage: UIImage?
    @State private var processedImage: UIImage?
    @State private var step = 0
    @State private var result: GlaucomaResult?
    @State private var errorMessage = ""

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
                        patient: patient,
                        image: processedImage,
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
                patientHeader

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

    private var patientHeader: some View {
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
                store.addRecord(for: patient, result: res, image: processed)
                self.processedImage = processed
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
        result = nil
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
