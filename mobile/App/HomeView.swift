import SwiftUI

// =====================================================================
//  HOME — landing tab after login. Quick actions (analysis without
//  picking/saving a patient first, new patient) plus a glance at recent
//  activity, so the doctor never has to dig for the thing they open the
//  app to do most often.
// =====================================================================

struct HomeView: View {
    @EnvironmentObject private var store: PatientStore
    @EnvironmentObject private var auth: AuthManager

    @State private var showQuickAnalysis = false
    @State private var showNewPatient = false
    @State private var newPatientForAnalysis: Patient?

    private var recentRecords: [AnalysisRecord] { Array(store.allRecordsByDate.prefix(3)) }

    var body: some View {
        ZStack {
            ScreenBackground()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 22) {
                    greeting

                    VStack(spacing: 12) {
                        Button {
                            showQuickAnalysis = true
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: "bolt.fill").font(.system(size: 18))
                                Text("Szybka analiza")
                            }
                        }
                        .buttonStyle(FilledButtonStyle())

                        Button {
                            showNewPatient = true
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: "person.fill.badge.plus").font(.system(size: 18))
                                Text("Nowy pacjent")
                            }
                        }
                        .buttonStyle(SoftButtonStyle(tint: .violet))
                    }

                    HStack(spacing: 14) {
                        StatCard(value: "\(store.patients.count)", label: "Pacjenci", icon: "person.2.fill", color: .brand)
                        StatCard(value: "\(store.records.count)", label: "Analizy", icon: "waveform.path.ecg", color: .violet)
                    }

                    if !recentRecords.isEmpty {
                        VStack(spacing: 14) {
                            SectionHeader(title: "Ostatnie analizy")

                            ForEach(recentRecords) { record in
                                if let patient = store.patient(by: record.patientId) {
                                    NavigationLink(value: record) {
                                        HistoryRow(record: record, patient: patient, image: store.image(for: record))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, DS.screenPadding)
                .padding(.top, 4)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Start")
        .navigationDestination(for: AnalysisRecord.self) { record in
            RecordDetailView(record: record, patient: store.patient(by: record.patientId))
        }
        .fullScreenCover(isPresented: $showQuickAnalysis) {
            AnalysisFlowView(patient: nil)
        }
        .sheet(isPresented: $showNewPatient) {
            PatientFormView { created in
                // Short delay avoids a presentation race between the
                // dismissing sheet and the incoming full-screen cover.
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    newPatientForAnalysis = created
                }
            }
        }
        .fullScreenCover(item: $newPatientForAnalysis) { patient in
            AnalysisFlowView(patient: patient)
        }
    }

    private var greeting: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(greetingText)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(Color.textPrimary)
            Text("Co dzisiaj robimy?")
                .font(.system(size: 14))
                .foregroundStyle(Color.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 4)
    }

    private var greetingText: String {
        guard auth.isLoggedIn, let username = auth.username else { return "Cześć!" }
        return "Cześć, \(username)"
    }
}
