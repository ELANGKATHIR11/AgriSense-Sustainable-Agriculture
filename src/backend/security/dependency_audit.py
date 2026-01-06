"""
Dependency Vulnerability Management
Tracks and manages security vulnerabilities in dependencies
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a security vulnerability in a package."""
    cve_id: str
    package_name: str
    affected_versions: List[str]
    fixed_version: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    cvss_score: float
    published_date: str
    references: List[str]


class DependencyAuditManager:
    """
    Manages dependency security audits and vulnerability tracking.
    """

    def __init__(self):
        self.vulnerabilities: Dict[str, List[Vulnerability]] = {}
        self.audit_results: List[Dict] = []
        self.last_audit_date: Optional[datetime] = None

    def add_vulnerability(self, vuln: Vulnerability) -> None:
        """Register a known vulnerability."""
        if vuln.package_name not in self.vulnerabilities:
            self.vulnerabilities[vuln.package_name] = []

        self.vulnerabilities[vuln.package_name].append(vuln)
        logger.warning(
            f"Registered vulnerability {vuln.cve_id} in {vuln.package_name} "
            f"(severity: {vuln.severity})"
        )

    def check_package_version(
        self, package_name: str, version: str
    ) -> Tuple[bool, List[Vulnerability]]:
        """
        Check if package version has known vulnerabilities.
        Returns: (is_safe, list_of_vulnerabilities)
        """
        if package_name not in self.vulnerabilities:
            return True, []

        affected_vulns = []
        for vuln in self.vulnerabilities[package_name]:
            if self._version_matches_affected(version, vuln.affected_versions):
                affected_vulns.append(vuln)

        return len(affected_vulns) == 0, affected_vulns

    @staticmethod
    def _version_matches_affected(version: str, affected_versions: List[str]) -> bool:
        """
        Check if version matches affected versions list.
        Handles ranges like "< 1.2.0", ">= 1.0.0", "== 1.1.5"
        """
        try:
            # Simple version comparison
            from packaging import version as pkg_version

            ver = pkg_version.parse(version)

            for spec in affected_versions:
                spec = spec.strip()

                if spec.startswith("<="):
                    max_ver = pkg_version.parse(spec[2:].strip())
                    if ver <= max_ver:
                        return True

                elif spec.startswith(">="):
                    min_ver = pkg_version.parse(spec[2:].strip())
                    if ver >= min_ver:
                        return True

                elif spec.startswith("<"):
                    max_ver = pkg_version.parse(spec[1:].strip())
                    if ver < max_ver:
                        return True

                elif spec.startswith(">"):
                    min_ver = pkg_version.parse(spec[1:].strip())
                    if ver > min_ver:
                        return True

                elif spec.startswith("=="):
                    target_ver = pkg_version.parse(spec[2:].strip())
                    if ver == target_ver:
                        return True

            return False

        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return False

    def audit_requirements(
        self, requirements: Dict[str, str]
    ) -> Dict[str, List[Vulnerability]]:
        """
        Audit all requirements for vulnerabilities.
        Returns: {package_name: [vulnerabilities]}
        """
        audit_results = {}

        for package_name, version in requirements.items():
            is_safe, vulns = self.check_package_version(package_name, version)

            if not is_safe:
                audit_results[package_name] = vulns
                logger.error(
                    f"Found {len(vulns)} vulnerabilities in {package_name}@{version}"
                )

        self.last_audit_date = datetime.utcnow()
        self.audit_results.append(
            {
                "timestamp": self.last_audit_date.isoformat(),
                "vulnerable_packages": len(audit_results),
                "results": audit_results,
            }
        )

        return audit_results

    def get_remediation_plan(
        self, audit_results: Dict[str, List[Vulnerability]]
    ) -> Dict[str, str]:
        """
        Generate remediation plan (suggested versions to upgrade to).
        Returns: {package_name: recommended_version}
        """
        plan = {}

        for package_name, vulns in audit_results.items():
            # Find highest fixed version
            fixed_versions = [v.fixed_version for v in vulns]
            if fixed_versions:
                plan[package_name] = max(fixed_versions)
                logger.info(
                    f"Recommend upgrade: {package_name} -> {plan[package_name]}"
                )

        return plan

    def get_severity_summary(
        self, audit_results: Dict[str, List[Vulnerability]]
    ) -> Dict[str, int]:
        """
        Get summary of vulnerabilities by severity.
        """
        summary = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

        for vulns in audit_results.values():
            for vuln in vulns:
                summary[vuln.severity] = summary.get(vuln.severity, 0) + 1

        return summary


# Known Python Dependencies Vulnerabilities
PYTHON_VULNERABILITIES = [
    Vulnerability(
        cve_id="CVE-2023-39611",
        package_name="urllib3",
        affected_versions=["<1.26.17", ">=2.0.0,<2.0.7"],
        fixed_version="2.0.7",
        severity="MEDIUM",
        description="Improper input validation in urlopen()",
        cvss_score=6.5,
        published_date="2023-09-08",
        references=[
            "https://github.com/urllib3/urllib3/security/advisories/GHSA-fr6q-rppq-8q3w"
        ],
    ),
    Vulnerability(
        cve_id="CVE-2024-27351",
        package_name="requests",
        affected_versions=["<2.32.0"],
        fixed_version="2.32.0",
        severity="MEDIUM",
        description="Unintended inclusion of Proxy-Authentication header in requests to proxies",
        cvss_score="5.9",
        published_date="2024-05-15",
        references=["https://github.com/psf/requests/security/advisories/GHSA-j8r2-6x86-q33q"],
    ),
    Vulnerability(
        cve_id="CVE-2023-5752",
        package_name="pip",
        affected_versions=["<23.3"],
        fixed_version="23.3",
        severity="MEDIUM",
        description="Cached HTTP request vulnerability",
        cvss_score="5.5",
        published_date="2023-11-09",
        references=["https://github.com/pypa/pip/security/advisories/GHSA-5pf6-fj9r-fqcv"],
    ),
    Vulnerability(
        cve_id="CVE-2024-3156",
        package_name="cryptography",
        affected_versions=["<42.0.0"],
        fixed_version="42.0.0",
        severity="HIGH",
        description="Potential issue with modular exponentiation in RSA",
        cvss_score="7.5",
        published_date="2024-03-01",
        references=["https://github.com/pyca/cryptography/security/advisories"],
    ),
]

# Known Node.js/JavaScript Vulnerabilities
JAVASCRIPT_VULNERABILITIES = [
    {
        "cve_id": "GHSA-q42p-pg78-2pp9",
        "package": "react",
        "affected_versions": ["<18.2.0"],
        "fixed_version": "18.2.0",
        "severity": "MEDIUM",
        "description": "Potential DOM clobbering in React",
        "published": "2023-10-17",
    },
    {
        "cve_id": "GHSA-8v57-r33f-f462",
        "package": "axios",
        "affected_versions": ["<1.6.0"],
        "fixed_version": "1.6.0",
        "severity": "HIGH",
        "description": "SSRF vulnerability in axios request validation",
        "published": "2023-09-29",
    },
    {
        "cve_id": "GHSA-5pf6-fj9r-fqcv",
        "package": "express",
        "affected_versions": ["<4.18.2"],
        "fixed_version": "4.18.2",
        "severity": "MEDIUM",
        "description": "Potential denial of service in Express",
        "published": "2023-10-26",
    },
]


class SecurityPatchTracker:
    """
    Track applied security patches and generate reports.
    """

    def __init__(self):
        self.applied_patches: List[Dict] = []

    def record_patch(
        self,
        cve_id: str,
        package_name: str,
        old_version: str,
        new_version: str,
        patch_date: str,
    ) -> None:
        """Record an applied security patch."""
        patch = {
            "cve_id": cve_id,
            "package": package_name,
            "old_version": old_version,
            "new_version": new_version,
            "patch_date": patch_date,
            "applied_at": datetime.utcnow().isoformat(),
        }
        self.applied_patches.append(patch)
        logger.info(f"Recorded patch for {cve_id}: {package_name} {old_version} -> {new_version}")

    def generate_patch_report(self) -> Dict:
        """Generate report of all applied patches."""
        return {
            "total_patches": len(self.applied_patches),
            "patches": self.applied_patches,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_patch_by_cve(self, cve_id: str) -> Optional[Dict]:
        """Look up patch by CVE ID."""
        for patch in self.applied_patches:
            if patch["cve_id"] == cve_id:
                return patch
        return None
