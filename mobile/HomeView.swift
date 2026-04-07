import SwiftUI
import PhotosUI

// Home Screen
struct HomeView: View {
    @ObservedObject var vm: GlaucomaViewModel
    @State private var appeared = false
    @State private var imageAppeared = false
    @State private var photosPickerItem: PhotosPickerItem?
    @State private var showSourceDialog = false
    @State private var showCamera = false
    @State private var showPhotosPicker = false

    var body: some View {
        ZStack {
            // Background
            Color.appBackground.ignoresSafeArea()

            // glow top center
            RadialGradient(
                colors: [Color.accentCyan.opacity(0.08), .clear],
                center: .top,
                startRadius: 0,
                endRadius: 360
            )
            .ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 0) {

                    // header:
                    VStack(spacing: 12) {
                        // eye
                        ZStack {
                            PulsingCircle(color: .accentCyan)
                                .frame(width: 88, height: 88)
                            Circle()
                                .stroke(Color.accentCyan.opacity(0.35), lineWidth: 1)
                                .frame(width: 80, height: 80)
                            Image(systemName: "eye.fill")
                                .font(.system(size: 34, weight: .light))
                                .foregroundStyle(Color.accentCyan)
                        }
                        .padding(.top, 52)

                        Text("JASKRA AI")
                            .font(.system(size: 28, weight: .bold, design: .monospaced))
                            .tracking(6)
                            .foregroundStyle(Color.textPrimary)

                        Text("Diagnostyka dna oka")
                            .font(.system(size: 14, weight: .medium))
                            .tracking(2)
                            .foregroundStyle(Color.textSecondary)
                            .textCase(.uppercase)
                    }
                    .opacity(appeared ? 1 : 0)
                    .offset(y: appeared ? 0 : -16)
                    .animation(.easeOut(duration: 0.7), value: appeared)

                    // image selection
                    Group {
                        if let img = vm.selectedImage {
                            // Selected image with change option
                            Button {
                                showSourceDialog = true
                            } label: {
                                Image(uiImage: img)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 260)
                                    .clipShape(RoundedRectangle(cornerRadius: 20))
                                    .glowBorder(Color.accentCyan.opacity(0.3), radius: 20)
                                    .overlay(
                                        VStack {
                                            Spacer()
                                            HStack {
                                                Spacer()
                                                Label("Zmień", systemImage: "arrow.triangle.2.circlepath")
                                                    .font(.system(size: 12, weight: .semibold))
                                                    .foregroundStyle(.white)
                                                    .padding(.horizontal, 12)
                                                    .padding(.vertical, 6)
                                                    .background(.ultraThinMaterial, in: Capsule())
                                                    .padding(14)
                                            }
                                        }
                                    )
                            }
                        } else {
                            // Photo source options
                            HStack(spacing: 16) {
                                Button {
                                    if UIImagePickerController.isSourceTypeAvailable(.camera) {
                                        showCamera = true
                                    } else {
                                        showPhotosPicker = true // Fallback to photo picker if camera is not available
                                    }
                                } label: {
                                    SourceCard(icon: "camera.fill", title: "Aparat", subtitle: "Zrób zdjęcie")
                                }

                                Button {
                                    showPhotosPicker = true
                                } label: {
                                    SourceCard(icon: "photo.on.rectangle.angled", title: "Galeria", subtitle: "Wybierz zdjęcie")
                                }
                            }
                            .frame(height: 160)
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 36)
                    .opacity(imageAppeared ? 1 : 0)
                    .offset(y: imageAppeared ? 0 : 20)
                    .animation(.easeOut(duration: 0.6).delay(0.25), value: imageAppeared)

                    // action button:
                    Button {
                        vm.runAnalysis()
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "waveform.path.ecg.rectangle.fill")
                                .font(.system(size: 18))
                            Text("ANALIZUJ")
                                .font(.system(size: 16, weight: .bold, design: .monospaced))
                                .tracking(3)
                        }
                        .foregroundStyle(vm.selectedImage != nil ? Color.appBackground : Color.textTertiary)
                        .frame(maxWidth: .infinity)
                        .frame(height: 56)
                        .background(vm.selectedImage != nil ? Color.accentCyan : Color.cardBorder)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .disabled(vm.selectedImage == nil)
                    .padding(.horizontal, 24)
                    .padding(.top, 20)
                    .opacity(imageAppeared ? 1 : 0)

                    // TODO: Time in info
                    HStack(spacing: 32) {
                        InfoPill(icon: "cpu", text: "AI Model")
                        InfoPill(icon: "lock.shield", text: "Prywatność")
                        InfoPill(icon: "bolt.fill", text: "~10 sek")
                    }
                    .padding(.top, 32)
                    .opacity(imageAppeared ? 1 : 0)

                    Text("Wynik ma charakter informacyjny i nie zastępuje konsultacji lekarskiej.")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.textTertiary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                        .padding(.top, 28)
                        .padding(.bottom, 48)
                        .opacity(imageAppeared ? 1 : 0)
                }
            }
        }
        .onAppear {
            appeared = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { imageAppeared = true }
        }
        .photosPicker(isPresented: $showPhotosPicker, selection: $photosPickerItem, matching: .images)
        
        .fullScreenCover(isPresented: $showCamera) {
            ImagePicker(image: $vm.selectedImage, sourceType: .camera)
                .ignoresSafeArea()
        }
        
        .confirmationDialog("Wybierz źródło", isPresented: $showSourceDialog) {
            Button("Aparat") { showCamera = true }
            Button("Galeria") { showPhotosPicker = true }
            Button("Anuluj", role: .cancel) {}
        }
        .onChange(of: photosPickerItem) { _, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    vm.selectedImage = uiImage
                }
            }
        }
    }
}

// Components for choice cards
struct SourceCard: View {
    let icon: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.accentCyan.opacity(0.1))
                    .frame(width: 50, height: 50)
                Image(systemName: icon)
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(Color.accentCyan)
            }
            VStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Color.textPrimary)
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundStyle(Color.textSecondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.cardBorder, style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
        )
    }
}

// Info Pill
struct InfoPill: View {
    let icon: String
    let text: String

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .light))
                .foregroundStyle(Color.accentCyan.opacity(0.7))
            Text(text)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.textTertiary)
                .tracking(0.5)
        }
    }
}