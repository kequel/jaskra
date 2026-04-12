import SwiftUI

// Home Screen
struct HomeView: View {
    @ObservedObject var vm: GlaucomaViewModel
    @State private var appeared = false
    @State private var imageAppeared = false

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
                        .opacity(appeared ? 1 : 0)
                        .offset(y: appeared ? 0 : -16)

                        Text("JASKRA AI")
                            .font(.system(size: 28, weight: .bold, design: .monospaced))
                            .tracking(6)
                            .foregroundStyle(Color.textPrimary)
                            .opacity(appeared ? 1 : 0)

                        Text("Diagnostyka dna oka")
                            .font(.system(size: 14, weight: .medium))
                            .tracking(2)
                            .foregroundStyle(Color.textSecondary)
                            .textCase(.uppercase)
                            .opacity(appeared ? 1 : 0)
                    }
                    .animation(.easeOut(duration: 0.7), value: appeared)

                    // image drop
                    Button {
                        vm.showPicker = true
                    } label: {
                        ZStack {
                            RoundedRectangle(cornerRadius: 20)
                                .fill(Color.cardBackground)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 20)
                                        .stroke(
                                            vm.selectedImage != nil
                                            ? Color.accentCyan.opacity(0.5)
                                            : Color.cardBorder,
                                            style: StrokeStyle(lineWidth: 1, dash: vm.selectedImage == nil ? [8, 5] : [])
                                        )
                                )
                                .shadow(color: vm.selectedImage != nil ? Color.accentCyan.opacity(0.15) : .clear,
                                        radius: 20)

                            if let img = vm.selectedImage {
                                // show selected image
                                Image(uiImage: img)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(height: 260)
                                    .clipShape(RoundedRectangle(cornerRadius: 20))
                                    .overlay(
                                        // tap to change hint
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
                            } else {
                                // empty state
                                VStack(spacing: 14) {
                                    ZStack {
                                        Circle()
                                            .fill(Color.accentCyan.opacity(0.08))
                                            .frame(width: 64, height: 64)
                                        Image(systemName: "photo.badge.plus")
                                            .font(.system(size: 26, weight: .light))
                                            .foregroundStyle(Color.accentCyan.opacity(0.7))
                                    }
                                    Text("Dotknij, aby wybrać zdjęcie")
                                        .font(.system(size: 15, weight: .medium))
                                        .foregroundStyle(Color.textSecondary)
                                    Text("Zdjęcie dna oka (fundus photography)")
                                        .font(.system(size: 12))
                                        .foregroundStyle(Color.textTertiary)
                                }
                                .frame(height: 200)
                            }
                        }
                        .frame(height: vm.selectedImage != nil ? 260 : 200)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 36)
                    .opacity(imageAppeared ? 1 : 0)
                    .offset(y: imageAppeared ? 0 : 20)
                    .animation(.easeOut(duration: 0.6).delay(0.25), value: imageAppeared)

                    // analyze button
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
                        .background(
                            vm.selectedImage != nil
                            ? Color.accentCyan
                            : Color.cardBorder
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .shadow(color: vm.selectedImage != nil ? Color.accentCyan.opacity(0.35) : .clear,
                                radius: 16, y: 4)
                    }
                    .disabled(vm.selectedImage == nil)
                    .padding(.horizontal, 24)
                    .padding(.top, 20)
                    .opacity(imageAppeared ? 1 : 0)
                    .animation(.easeOut(duration: 0.5).delay(0.4), value: imageAppeared)

                    // TODO: Time in info
                    HStack(spacing: 32) {
                        InfoPill(icon: "cpu", text: "AI Model")
                        InfoPill(icon: "lock.shield", text: "Prywatność")
                        InfoPill(icon: "bolt.fill", text: "~10 sek")
                    }
                    .padding(.top, 32)
                    .opacity(imageAppeared ? 1 : 0)
                    .animation(.easeOut(duration: 0.5).delay(0.55), value: imageAppeared)

                    //  Disclaimer
                    Text("Wynik ma charakter informacyjny i nie zastępuje konsultacji lekarskiej.")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.textTertiary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                        .padding(.top, 28)
                        .padding(.bottom, 48)
                        .opacity(imageAppeared ? 1 : 0)
                        .animation(.easeOut(duration: 0.5).delay(0.65), value: imageAppeared)
                }
            }
        }
        .onAppear {
            appeared = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { imageAppeared = true }
        }
        .sheet(isPresented: $vm.showPicker) {
            ImagePicker(image: $vm.selectedImage)
        }
    }
}

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
