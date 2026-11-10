"""Tests for patient/doctor registration and administrator verification."""

import asyncio
import unittest
import sys
import types
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException, Request

import app as api
from src.auth_service import hash_password
from src.database import (
    Base,
    DoctorProfile,
    PatientProfile,
    ScreeningResponse,
    ScreeningResult,
    ScreeningSession,
    User,
)


class RoleAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )
        self.db = session_factory()
        self.admin = User(
            full_name="Test Admin",
            email="admin@example.com",
            password_hash=hash_password("admin-password"),
            role="admin",
            account_status="active",
        )
        self.db.add(self.admin)
        self.db.commit()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    @staticmethod
    def request(path):
        return Request({
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": [],
        })

    def test_root_requires_login_and_screening_is_a_separate_route(self):
        root_response = api.frontend_root(self.request("/"))
        self.assertEqual(root_response.status_code, 302)
        self.assertEqual(root_response.headers["location"], "/auth/login")

        screening_response = api.screening_page(self.request("/screening"))
        self.assertEqual(screening_response.status_code, 302)
        self.assertEqual(screening_response.headers["location"], "/auth/login")

        login_page = api.auth_login(self.request("/auth/login"))
        self.assertEqual(login_page.path.name, "login.html")

    def test_google_sign_in_is_optional_and_links_only_verified_existing_email(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(api.google_auth_status()["configured"])

        with (
            patch.dict(
                "os.environ",
                {
                    "GOOGLE_CLIENT_ID": "test-client",
                    "GOOGLE_CLIENT_SECRET": "test-secret",
                },
            ),
            patch.object(api, "SessionLocal", return_value=self.db),
        ):
            with patch.object(
                api.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "google-user-id",
                        "email": "google-patient@example.com",
                        "email_verified": True,
                    },
                }),
            ):
                unregistered = asyncio.run(
                    api.google_callback(self.request("/auth/google/callback"))
                )
            self.assertEqual(
                unregistered.headers["location"],
                "/auth/login?google_error=account_not_found",
            )

            with patch.object(
                api.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "unverified-google-user",
                        "email": "unverified@example.com",
                        "email_verified": False,
                    },
                }),
            ):
                unverified = asyncio.run(
                    api.google_callback(self.request("/auth/google/callback"))
                )
            self.assertEqual(
                unverified.headers["location"],
                "/auth/login?google_error=email_not_verified",
            )

            pending_doctor = api.register_user(
                api.RegisterRequest(
                    full_name="Pending Doctor",
                    email="pending-google-doctor@example.com",
                    password="doctor-password",
                    confirm_password="doctor-password",
                    role="doctor",
                    consent=True,
                    medical_registration_number="REG-PENDING-GOOGLE",
                    qualification="MBBS",
                    specialization="Obstetrics",
                    hospital="Test Clinic",
                    years_of_experience=4,
                ),
                self.db,
            )
            with patch.object(
                api.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "pending-doctor-google-user",
                        "email": "pending-google-doctor@example.com",
                        "email_verified": True,
                    },
                }),
            ):
                doctor_login = asyncio.run(
                    api.google_callback(self.request("/auth/google/callback"))
                )
            self.assertEqual(
                doctor_login.headers["location"],
                "/auth/login?google_error=doctor_pending",
            )

            registered = api.register_user(
                api.RegisterRequest(
                    full_name="Google Patient",
                    email="google-patient@example.com",
                    password="patient-password",
                    confirm_password="patient-password",
                    consent=True,
                ),
                self.db,
            )
            with patch.object(
                api.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "google-user-id",
                        "email": "google-patient@example.com",
                        "email_verified": True,
                    },
                }),
            ):
                response = asyncio.run(
                    api.google_callback(self.request("/auth/google/callback"))
                )

        self.assertEqual(registered["user"]["role"], "patient")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/patient")
        self.assertIn("sneha_session=", response.headers["set-cookie"])
        self.assertEqual(
            self.db.query(User).filter_by(
                id=registered["user"]["id"]
            ).one().google_subject,
            "google-user-id",
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "GOOGLE_CLIENT_ID": "test-client",
                    "GOOGLE_CLIENT_SECRET": "test-secret",
                },
            ),
            patch.object(api, "SessionLocal", return_value=self.db),
            patch.object(
                api.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "google-user-id",
                        "email": "updated-google-email@example.com",
                        "email_verified": True,
                    },
                }),
            ),
        ):
            linked_login = asyncio.run(
                api.google_callback(self.request("/auth/google/callback"))
            )
        self.assertEqual(linked_login.headers["location"], "/patient")

    def test_patient_registration_saves_patient_profile_details(self):
        payload = api.RegisterRequest(
            full_name="Test Patient",
            email="patient@example.com",
            password="patient-password",
            confirm_password="patient-password",
            role="patient",
            consent=True,
            phone="5550100",
            date_of_birth="1998-05-04",
            gender="female",
            address="Test address",
        )

        result = api.register_user(payload, self.db)

        profile = self.db.query(PatientProfile).filter_by(
            user_id=result["user"]["id"]
        ).one()
        self.assertEqual(profile.date_of_birth.isoformat(), "1998-05-04")
        self.assertEqual(profile.gender, "female")
        self.assertEqual(profile.contact_phone, "5550100")
        patient = self.db.query(User).filter_by(
            id=result["user"]["id"]
        ).one()
        updated = api.update_patient_profile(
            api.PatientProfileUpdate(gestational_age=22.5, city="Test City"),
            patient,
            self.db,
        )
        self.assertEqual(updated["profile"]["gestational_age"], 22.5)
        self.assertEqual(updated["profile"]["city"], "Test City")

    def test_doctor_requires_approval_before_login(self):
        payload = api.RegisterRequest(
            full_name="Test Doctor",
            email="doctor@example.com",
            password="doctor-password",
            confirm_password="doctor-password",
            role="doctor",
            consent=True,
            phone="5550101",
            medical_registration_number="REG-TEST-100",
            qualification="MBBS",
            specialization="Obstetrics",
            hospital="Test Clinic",
            years_of_experience=4,
        )

        result = api.register_user(payload, self.db)
        doctor = self.db.query(User).filter_by(id=result["user"]["id"]).one()
        profile = self.db.query(DoctorProfile).filter_by(user_id=doctor.id).one()

        self.assertEqual(doctor.account_status, "pending_verification")
        self.assertEqual(profile.verification_status, "pending")
        with self.assertRaises(HTTPException) as context:
            api.login_user(
                api.LoginRequest(
                    email="doctor@example.com",
                    password="doctor-password",
                ),
                api.Response(),
                self.db,
            )
        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("awaiting administrator verification", context.exception.detail)

        api.update_doctor_verification(
            doctor.id,
            api.DoctorVerificationRequest(verification_status="verified"),
            self.admin,
            self.db,
        )
        response = api.Response()
        login = api.login_user(
            api.LoginRequest(
                email="doctor@example.com",
                password="doctor-password",
            ),
            response,
            self.db,
        )

        self.assertEqual(login["redirect_url"], "/doctor")
        self.assertIn("set-cookie", response.headers)
        self.assertEqual(profile.verification_status, "verified")

    def test_duplicate_medical_registration_number_is_rejected(self):
        for email in ("doctor1@example.com", "doctor2@example.com"):
            payload = api.RegisterRequest(
                full_name="Test Doctor",
                email=email,
                password="doctor-password",
                confirm_password="doctor-password",
                role="doctor",
                consent=True,
                medical_registration_number="REG-DUPLICATE",
                qualification="MBBS",
                specialization="Obstetrics",
                hospital="Test Clinic",
                years_of_experience=2,
            )
            if email.endswith("doctor1@example.com"):
                api.register_user(payload, self.db)
            else:
                with self.assertRaises(HTTPException) as context:
                    api.register_user(payload, self.db)
                self.assertEqual(context.exception.status_code, 409)

        self.assertEqual(
            self.db.query(User).filter(User.role == "doctor").count(),
            1,
        )

    def test_patient_assessment_saves_answers_and_server_prediction(self):
        registration = api.RegisterRequest(
            full_name="Assessment Test Patient",
            email="assessment-patient@example.com",
            password="patient-password",
            confirm_password="patient-password",
            consent=True,
        )
        result = api.register_user(registration, self.db)
        patient = self.db.query(User).filter_by(id=result["user"]["id"]).one()

        from src.sneha.feature_extractor import CATEGORY_VALUES
        from src.sneha.feature_schema import (
            CATEGORICAL_FEATURES,
            FEATURE_SCHEMA,
            NUMERIC_FEATURES,
        )

        answers = {}
        numeric_defaults = {
            "Age": 25,
            "Gestational Age": 20,
            "Number of sons": 0,
            "Number of daughters": 0,
            "Total Number of Children": 0,
        }
        for feature in FEATURE_SCHEMA:
            if feature in NUMERIC_FEATURES:
                answers[feature] = numeric_defaults[feature]
            elif feature in CATEGORICAL_FEATURES:
                answers[feature] = next(
                    item
                    for item in CATEGORY_VALUES[feature]
                    if item != "__MISSING__"
                )
            else:
                answers[feature] = 0

        predictor = types.ModuleType("src.predict_transformer")
        predictor.predict_antenatal_depression = lambda _answers: {
            "label": "Depressed",
            "probability": 0.71,
        }
        predictor.get_model_info = lambda: {
            "model": "Test transformer",
            "version": "test",
        }
        with patch.dict(sys.modules, {"src.predict_transformer": predictor}):
            response = api.predict_from_form(
                api.ScreeningFormSubmission(answers=answers),
                patient,
                self.db,
            )

        session_id = response["saved_session_id"]
        screening = self.db.query(ScreeningSession).filter_by(id=session_id).one()
        stored_answers = self.db.query(ScreeningResponse).filter_by(
            screening_session_id=session_id
        ).all()
        stored_result = self.db.query(ScreeningResult).filter_by(
            screening_session_id=session_id
        ).one()
        profile = self.db.query(PatientProfile).filter_by(
            user_id=patient.id
        ).one()

        self.assertEqual(screening.patient_id, profile.id)
        self.assertEqual(len(stored_answers), len(FEATURE_SCHEMA))
        self.assertEqual(stored_result.predicted_class, "Depressed")
        self.assertAlmostEqual(stored_result.risk_score, 0.71)
        self.assertIn("not a clinical diagnosis", stored_result.screening_summary)

    def test_doctor_access_requires_consent_and_revocation_takes_effect(self):
        patient_result = api.register_user(
            api.RegisterRequest(
                full_name="Shared Test Patient",
                email="shared-patient@example.com",
                password="patient-password",
                confirm_password="patient-password",
                consent=True,
            ),
            self.db,
        )
        patient = self.db.query(User).filter_by(
            id=patient_result["user"]["id"]
        ).one()
        patient_profile = self.db.query(PatientProfile).filter_by(
            user_id=patient.id
        ).one()

        def create_verified_doctor(email, registration_number):
            result = api.register_user(
                api.RegisterRequest(
                    full_name="Test Doctor",
                    email=email,
                    password="doctor-password",
                    confirm_password="doctor-password",
                    role="doctor",
                    consent=True,
                    medical_registration_number=registration_number,
                    qualification="MBBS",
                    specialization="Obstetrics",
                    hospital="Test Clinic",
                    years_of_experience=4,
                ),
                self.db,
            )
            doctor_user = self.db.query(User).filter_by(
                id=result["user"]["id"]
            ).one()
            api.update_doctor_verification(
                doctor_user.id,
                api.DoctorVerificationRequest(
                    verification_status="verified"
                ),
                self.admin,
                self.db,
            )
            return doctor_user

        doctor = create_verified_doctor("assigned-doctor@example.com", "REG-ASSIGNED")
        other_doctor = create_verified_doctor("other-doctor@example.com", "REG-OTHER")
        session_id = api.persist_patient_assessment(
            db=self.db,
            current_user=patient,
            answers={"Age": 28, "Gestational Age": 24},
            screening_status="completed",
            predicted_class="Not Depressed",
            risk_score=0.1,
            model_name="Test",
            model_version="test",
        )
        screening = self.db.query(ScreeningSession).filter_by(id=session_id).one()

        with self.assertRaises(HTTPException) as context:
            api.doctor_patient_history(patient_profile.id, other_doctor, self.db)
        self.assertEqual(context.exception.status_code, 404)

        api.share_patient_record_with_doctor(doctor.id, patient, self.db)
        doctor_dashboard = api.doctor_dashboard(doctor, self.db)
        doctor_patients = api.list_doctor_patients("", doctor, self.db)
        self.assertEqual(doctor_dashboard["metrics"]["assigned_patients"], 1)
        self.assertEqual(doctor_patients["patients"][0]["patient_id"], patient_profile.id)
        note = api.add_clinical_note(
            patient_profile.id,
            api.ClinicalNoteRequest(
                note="Test clinical observation",
                screening_session_id=screening.id,
            ),
            doctor,
            self.db,
        )
        recommendation = api.add_patient_recommendation(
            patient_profile.id,
            api.RecommendationRequest(
                recommendation="Discuss follow-up with the care team",
                screening_session_id=screening.id,
            ),
            doctor,
            self.db,
        )
        history = api.doctor_patient_history(
            patient_profile.id,
            doctor,
            self.db,
        )
        self.assertEqual(history["patient"]["full_name"], "Shared Test Patient")
        self.assertEqual(len(history["screenings"]), 1)
        self.assertEqual(history["clinical_notes"][0]["id"], note["id"])
        self.assertEqual(
            history["recommendations"][0]["id"],
            recommendation["id"],
        )
        patient_dashboard = api.patient_dashboard(patient, self.db)
        self.assertEqual(len(patient_dashboard["clinical_notes"]), 1)
        self.assertEqual(len(patient_dashboard["recommendations"]), 1)

        api.revoke_patient_record_consent(doctor.id, patient, self.db)
        with self.assertRaises(HTTPException) as context:
            api.doctor_patient_history(patient_profile.id, doctor, self.db)
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
