#!/usr/bin/env python3
"""
OpenStack Instance Manager for Running Unit Tests
Automatically creates instances, runs tests, reports results, and cleans up.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from openstack import connection


class OpenStackTestRunner:
    """Manages OpenStack instances for running unit tests"""
    
    def __init__(self):
        """Initialize OpenStack connection using environment variables"""
        try:
            # Read environment variables set by openstack-rc.sh
            auth_url = os.environ.get('OS_AUTH_URL')
            username = os.environ.get('OS_USERNAME')
            password = os.environ.get('OS_PASSWORD')
            project_name = os.environ.get('OS_PROJECT_NAME')
            user_domain_name = os.environ.get('OS_USER_DOMAIN_NAME', 'Default')
            project_domain_name = os.environ.get('OS_PROJECT_DOMAIN_NAME', 'Default')
            region_name = os.environ.get('OS_REGION_NAME', 'RegionOne')
            
            if not all([auth_url, username, password, project_name]):
                print("✗ Missing required environment variables")
                raise Exception("Missing OpenStack credentials")
            
            self.conn = connection.Connection(
                auth_url=auth_url,
                username=username,
                password=password,
                project_name=project_name,
                user_domain_name=user_domain_name,
                project_domain_name=project_domain_name,
                region_name=region_name,
                auth_type='password'
            )
            print("✓ Connected to OpenStack")
            
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            raise
        
        self.instances = []
        self.test_results = {}
        
    def create_instance(self, instance_name, image_name, flavor_name, network_name=None):
        """Create a new OpenStack instance"""
        try:
            print(f"\nCreating instance: {instance_name}")
            
            image = self.conn.compute.find_image(image_name)
            if not image:
                raise Exception(f"Image '{image_name}' not found")
            
            flavor = self.conn.compute.find_flavor(flavor_name)
            if not flavor:
                raise Exception(f"Flavor '{flavor_name}' not found")
            
            server = self.conn.compute.create_server(
                name=instance_name,
                image_id=image.id,
                flavor_id=flavor.id,
                networks=[{"name": network_name}] if network_name else None
            )
            
            server = self._wait_for_instance(server)
            self.instances.append(server)
            print(f"✓ Instance created with ID: {server.id}")
            return server
            
        except Exception as e:
            print(f"✗ Error creating instance: {e}")
            raise
    
    def _wait_for_instance(self, server, timeout=300, interval=5):
        """Wait for instance to reach ACTIVE state"""
        print("  Waiting for instance to start...")
        start = time.time()
        while time.time() - start < timeout:
            server = self.conn.compute.get_server(server.id)
            
            if server.status == "ACTIVE":
                print(f"  ✓ Instance is active")
                return server
            elif server.status in ["ERROR", "DELETED"]:
                raise Exception(f"Instance failed: {server.status}")
            
            time.sleep(interval)
        
        raise TimeoutError(f"Instance startup timed out after {timeout}s")
    
    def get_instance_ip(self, server):
        """Get the IP address of an instance"""
        server = self.conn.compute.get_server(server.id)
        
        # Try floating IP first
        try:
            for floating_ip in self.conn.network.ips():
                if floating_ip.port_id:
                    port = self.conn.network.get_port(floating_ip.port_id)
                    if port.device_id == server.id:
                        return str(floating_ip.floating_ip_address)
        except:
            pass
        
        # Fall back to fixed IP
        if server.addresses:
            for network_name, addresses in server.addresses.items():
                for addr in addresses:
                    return addr['addr']
        
        return None
    
    def download_model_files(self, server, github_repo, release_tag, ssh_user="ubuntu"):
        """Download model files from GitHub release"""
        ip = self.get_instance_ip(server)
        if not ip:
            raise Exception("Could not get IP for instance")
        
        print(f"  Downloading model files from release: {release_tag}")
        
        # Create tests/models directory
        mkdir_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ssh_user}@{ip} 'mkdir -p ~/animl/tests/models'"
        subprocess.run(mkdir_cmd, shell=True, check=True, capture_output=True, timeout=30)
        
        # Download all .pt and .csv files from release
        download_cmd = f"""
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ssh_user}@{ip} '
        cd ~/animl/tests/models && 
        wget -q https://github.com/{github_repo}/releases/download/{release_tag}/*.pt 2>/dev/null || true &&
        wget -q https://github.com/{github_repo}/releases/download/{release_tag}/*.csv 2>/dev/null || true
        '
        """
        
        result = subprocess.run(download_cmd, shell=True, capture_output=True, timeout=120, text=True)
        
        if result.returncode != 0 and result.stderr:
            print(f"  ⚠️  Warning: Model download returned exit code {result.returncode}")
            print(f"  Details: {result.stderr}")
        else:
            print(f"  ✓ Model files downloaded")
    
    def run_tests_on_instance(self, server, repo_url, github_repo="conservationtechlab/animl-py", release_tag="sdzwa_southwest_v3", ssh_user="ubuntu"):
        """SSH into instance, setup environment, and run unit tests"""
        ip = self.get_instance_ip(server)
        if not ip:
            raise Exception(f"Could not get IP for instance")
        
        print(f"\nRunning tests on {server.name} ({ip})")
        self._wait_for_ssh(ip)
        
        try:
            # Clone repository
            print(f"  Cloning repository...")
            clone_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ssh_user}@{ip} 'git clone {repo_url} ~/animl'"
            subprocess.run(clone_cmd, shell=True, check=True, capture_output=True, timeout=120)
            
            # Download model files from GitHub release
            self.download_model_files(server, github_repo, release_tag, ssh_user)
            
            # Install requirements
            print(f"  Installing requirements...")
            install_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ssh_user}@{ip} 'cd ~/animl && pip install -r requirements.txt'"
            result = subprocess.run(
                install_cmd,
                shell=True,
                capture_output=True,
                timeout=300,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  ⚠️  Warning: pip install returned exit code {result.returncode}")
                if result.stderr:
                    print(f"  Details: {result.stderr}")
            
            # Run tests
            print(f"  Running unit tests...")
            test_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ssh_user}@{ip} 'cd ~/animl && python -m unittest discover -s tests -p \"test_*.py\" -v'"
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                timeout=600,
                text=True
            )
            
            test_result = {
                "instance": server.name,
                "instance_id": server.id,
                "ip": ip,
                "timestamp": datetime.now().isoformat(),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            self.test_results[server.name] = test_result
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ Tests timed out")
            return {
                "instance": server.name,
                "instance_id": server.id,
                "ip": ip,
                "timestamp": datetime.now().isoformat(),
                "error": "Test execution timed out",
                "success": False
            }
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {
                "instance": server.name,
                "instance_id": server.id,
                "ip": ip,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
    def _wait_for_ssh(self, ip, timeout=120, interval=5):
        """Wait for SSH to be available"""
        print(f"  Waiting for SSH...")
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                f"nc -z -w 5 {ip} 22",
                shell=True,
                capture_output=True
            )
            if result.returncode == 0:
                print(f"  ✓ SSH ready")
                time.sleep(5)
                return
            time.sleep(interval)
        
        raise TimeoutError(f"SSH unavailable after {timeout}s")
    
    def generate_report(self):
        """Generate test results report"""
        report = []
        report.append("=" * 80)
        report.append("ANIML UNIT TEST RESULTS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r.get("success"))
        failed = total - passed
        
        report.append(f"SUMMARY")
        report.append(f"  Total Test Runs: {total}")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {failed}")
        if total > 0:
            report.append(f"  Success Rate: {(passed/total*100):.1f}%")
        report.append("")
        
        for instance_name, result in self.test_results.items():
            report.append("-" * 80)
            report.append(f"Instance: {instance_name}")
            report.append(f"Instance ID: {result.get('instance_id')}")
            report.append(f"IP Address: {result.get('ip')}")
            report.append(f"Test Time: {result.get('timestamp')}")
            
            if result.get('success'):
                report.append("Status: ✓ PASSED")
            else:
                report.append("Status: ✗ FAILED")
            
            if "error" in result:
                report.append(f"Error: {result['error']}")
            else:
                report.append(f"Exit Code: {result.get('return_code')}")
                
            if result.get('stdout'):
                report.append(f"\nTest Output:")
                report.append("-" * 40)
                report.append(result['stdout'])
                
            if result.get('stderr'):
                report.append(f"\nStderr:")
                report.append("-" * 40)
                report.append(result['stderr'])
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_report(self, filename="test_report.txt"):
        """Save report to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {filename}")
    
    def cleanup(self):
        """Delete all instances"""
        print("\n" + "=" * 80)
        print("CLEANING UP")
        print("=" * 80)
        
        for server in self.instances:
            try:
                print(f"  Deleting: {server.name}")
                self.conn.compute.delete_server(server.id)
                print(f"  ✓ Deleted")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    def list_available_images(self):
        """List available images"""
        print("\n📦 Available Images:")
        for image in self.conn.compute.images():
            print(f"  - {image.name}")
    
    def list_available_flavors(self):
        """List available flavors"""
        print("\n⚙️  Available Flavors:")
        for flavor in self.conn.compute.flavors():
            print(f"  - {flavor.name} ({flavor.vcpus}vCPU, {flavor.ram}MB RAM, {flavor.disk}GB disk)")
    
    def list_available_networks(self):
        """List available networks"""
        print("\n🌐 Available Networks:")
        for network in self.conn.network.networks():
            print(f"  - {network.name}")


def main():
    """Main execution"""
    
    # ===== CONFIGURE THESE VALUES =====
    IMAGE = "Ubuntu 20.04"                                      # Your image name
    FLAVOR = "m1.small"                                         # Your flavor
    NETWORK = "private"                                         # Your network
    REPO_URL = "https://github.com/conservationtechlab/animl-py.git"
    GITHUB_REPO = "conservationtechlab/animl-py"
    RELEASE_TAG = "sdzwa_southwest_v3"
    SSH_USER = "ubuntu"
    # ==================================
    
    runner = OpenStackTestRunner()
    
    # Show available resources
    if len(sys.argv) > 1 and sys.argv[1] == '--discover':
        runner.list_available_images()
        runner.list_available_flavors()
        runner.list_available_networks()
        return
    
    try:
        # Create instance
        server = runner.create_instance(
            instance_name="animl-test-01",
            image_name=IMAGE,
            flavor_name=FLAVOR,
            network_name=NETWORK
        )
        
        # Run tests (clones repo, downloads models, installs requirements, runs tests)
        result = runner.run_tests_on_instance(
            server,
            repo_url=REPO_URL,
            github_repo=GITHUB_REPO,
            release_tag=RELEASE_TAG,
            ssh_user=SSH_USER
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Status: {'✓ PASSED' if result.get('success') else '✗ FAILED'}")
        if result.get('stdout'):
            print(f"\n{result['stdout']}")
        if result.get('stderr'):
            print(f"\nStderr:\n{result['stderr']}")
        
        # Save report
        runner.save_report()
        
    finally:
        # Always cleanup
        runner.cleanup()


if __name__ == "__main__":
    main()
