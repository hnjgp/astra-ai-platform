from tools.system import get_system_status


result = get_system_status()


assert result["service"] == "Astra"
assert result["status"] == "healthy"


print("TEST: System Tool PASS")