import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires running services - use docker-test")
class TestAPIIntegration:

    @pytest.fixture
    def api_client(self):
        from src.api.main import app
        return TestClient(app)

    def test_health_check(self, api_client):
        response = api_client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "process_menu" in data["endpoints"]

    @patch("src.api.services.mcp_client.MCPClient.classify_and_calculate")
    def test_process_menu_base64_mocked(self, mock_classify, api_client):
        mock_classify.return_value = AsyncMock(return_value={
            "status": "success",
            "vegetarian_items": [{"name": "Greek Salad", "price": 8.5}],
            "total_sum": 8.5
        })()
        
        from PIL import Image
        import io
        import base64
        
        img = Image.new("RGB", (100, 50), color="white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Greek Salad $8.50", fill="black")
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()
        
        response = api_client.post(
            "/process-menu/base64",
            json={"images": [b64_image]}
        )
        
        assert response.status_code == 200


@pytest.mark.skip(reason="Requires running services - use docker-test")
class TestMCPIntegration:

    @pytest.fixture
    def mcp_client(self):
        from src.mcp.main import app
        return TestClient(app)

    def test_mcp_health_check(self, mcp_client):
        response = mcp_client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_search_knowledge_base(self, mcp_client):
        response = mcp_client.get("/tools/search?query=tofu&top_k=3")
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 3
