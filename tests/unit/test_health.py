"""
Unit tests for health check endpoints and utilities.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.v1.health import (
    router,
    health_check,
    detailed_health_check,
    readiness_check,
    liveness_check,
    check_database_quick,
    check_redis_quick
)


class TestBasicHealthCheck:
    """Test basic health check endpoint"""
    
    def test_health_check_success(self):
        """Test basic health check returns success"""
        result = asyncio.run(health_check())
        
        assert result["status"] == "healthy"
        assert result["service"] == "autocare-backend"
        assert "timestamp" in result
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(result["timestamp"])
        assert isinstance(timestamp, datetime)
    
    def test_health_check_response_structure(self):
        """Test health check response has correct structure"""
        result = asyncio.run(health_check())
        
        expected_keys = {"status", "timestamp", "service"}
        assert set(result.keys()) == expected_keys


class TestDetailedHealthCheck:
    """Test detailed health check endpoint"""
    
    @pytest.fixture
    def mock_db_handler(self):
        """Mock database handler"""
        handler = MagicMock()
        handler.health_check = AsyncMock()
        handler.get_connection_stats = AsyncMock()
        return handler
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Mock Redis cache"""
        cache = MagicMock()
        cache.health_check = AsyncMock()
        cache.get_stats = AsyncMock()
        return cache
    
    def test_detailed_health_check_all_healthy(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check when all services are healthy"""
        # Setup mocks
        mock_db_handler.health_check.return_value = True
        mock_db_handler.get_connection_stats.return_value = {
            "active_connections": 5,
            "pool_size": 10
        }
        
        mock_redis_cache.health_check.return_value = True
        mock_redis_cache.get_stats.return_value = {
            "connected_clients": 3,
            "used_memory": 1024000
        }
        
        result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        assert result["status"] == "healthy"
        assert result["service"] == "autocare-backend"
        assert result["version"] == "1.0.0"
        assert "response_time_ms" in result
        
        # Check database status
        assert result["checks"]["database"]["status"] == "healthy"
        assert result["checks"]["database"]["stats"]["active_connections"] == 5
        
        # Check Redis status
        assert result["checks"]["redis"]["status"] == "healthy"
        assert result["checks"]["redis"]["stats"]["connected_clients"] == 3
    
    def test_detailed_health_check_database_unhealthy(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check with unhealthy database"""
        # Setup mocks
        mock_db_handler.health_check.return_value = False
        mock_redis_cache.health_check.return_value = True
        mock_redis_cache.get_stats.return_value = {"connected_clients": 3}
        
        result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        assert result["status"] == "degraded"
        assert result["checks"]["database"]["status"] == "unhealthy"
        assert result["checks"]["redis"]["status"] == "healthy"
    
    def test_detailed_health_check_database_exception(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check with database exception"""
        # Setup mocks
        mock_db_handler.health_check.side_effect = Exception("Connection failed")
        mock_redis_cache.health_check.return_value = True
        mock_redis_cache.get_stats.return_value = {"connected_clients": 3}
        
        with patch('app.api.v1.health.logger') as mock_logger:
            mock_logger.error = AsyncMock()
            
            result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
            
            assert result["status"] == "degraded"
            assert result["checks"]["database"]["status"] == "unhealthy"
            assert result["checks"]["database"]["error"] == "Connection failed"
            mock_logger.error.assert_called_once()
    
    def test_detailed_health_check_redis_unhealthy(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check with unhealthy Redis"""
        # Setup mocks
        mock_db_handler.health_check.return_value = True
        mock_db_handler.get_connection_stats.return_value = {"active_connections": 5}
        mock_redis_cache.health_check.return_value = False
        
        result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        assert result["status"] == "degraded"
        assert result["checks"]["database"]["status"] == "healthy"
        assert result["checks"]["redis"]["status"] == "unhealthy"
    
    def test_detailed_health_check_redis_exception(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check with Redis exception"""
        # Setup mocks
        mock_db_handler.health_check.return_value = True
        mock_db_handler.get_connection_stats.return_value = {"active_connections": 5}
        mock_redis_cache.health_check.side_effect = Exception("Redis connection failed")
        
        with patch('app.api.v1.health.logger') as mock_logger:
            mock_logger.error = AsyncMock()
            
            result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
            
            assert result["status"] == "degraded"
            assert result["checks"]["redis"]["status"] == "unhealthy"
            assert result["checks"]["redis"]["error"] == "Connection failed"
            mock_logger.error.assert_called_once()
    
    def test_detailed_health_check_all_unhealthy(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check when all services are unhealthy"""
        # Setup mocks
        mock_db_handler.health_check.return_value = False
        mock_redis_cache.health_check.return_value = False
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["status"] == "unhealthy"
    
    def test_detailed_health_check_response_time(self, mock_db_handler, mock_redis_cache):
        """Test detailed health check includes response time"""
        # Setup mocks
        mock_db_handler.health_check.return_value = True
        mock_db_handler.get_connection_stats.return_value = {}
        mock_redis_cache.health_check.return_value = True
        mock_redis_cache.get_stats.return_value = {}
        
        result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        assert "response_time_ms" in result
        assert isinstance(result["response_time_ms"], (int, float))
        assert result["response_time_ms"] >= 0


class TestReadinessCheck:
    """Test readiness probe endpoint"""
    
    def test_readiness_check_success(self):
        """Test readiness check when services are ready"""
        with patch('app.api.v1.health.check_database_quick', new_callable=AsyncMock) as mock_db_check:
            with patch('app.api.v1.health.check_redis_quick', new_callable=AsyncMock) as mock_redis_check:
                mock_db_check.return_value = True
                mock_redis_check.return_value = True
                
                result = asyncio.run(readiness_check())
                
                assert result["status"] == "ready"
                assert "timestamp" in result
    
    def test_readiness_check_database_not_ready(self):
        """Test readiness check when database is not ready"""
        with patch('app.api.v1.health.check_database_quick', new_callable=AsyncMock) as mock_db_check:
            with patch('app.api.v1.health.check_redis_quick', new_callable=AsyncMock) as mock_redis_check:
                mock_db_check.return_value = False
                mock_redis_check.return_value = True
                
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(readiness_check())
                
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["status"] == "not ready"
    
    def test_readiness_check_redis_not_ready(self):
        """Test readiness check when Redis is not ready"""
        with patch('app.api.v1.health.check_database_quick', new_callable=AsyncMock) as mock_db_check:
            with patch('app.api.v1.health.check_redis_quick', new_callable=AsyncMock) as mock_redis_check:
                mock_db_check.return_value = True
                mock_redis_check.return_value = False
                
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(readiness_check())
                
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["status"] == "not ready"
    
    def test_readiness_check_exception(self):
        """Test readiness check with exception"""
        with patch('app.api.v1.health.check_database_quick', new_callable=AsyncMock) as mock_db_check:
            mock_db_check.side_effect = Exception("Service check failed")
            
            with patch('app.api.v1.health.logger') as mock_logger:
                mock_logger.error = AsyncMock()
                
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(readiness_check())
                
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["status"] == "not ready"
                assert "Service check failed" in exc_info.value.detail["error"]


class TestLivenessCheck:
    """Test liveness probe endpoint"""
    
    def test_liveness_check_success(self):
        """Test liveness check always returns alive"""
        result = asyncio.run(liveness_check())
        
        assert result["status"] == "alive"
        assert "timestamp" in result
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(result["timestamp"])
        assert isinstance(timestamp, datetime)


class TestQuickChecks:
    """Test quick check utility functions"""
    
    def test_check_database_quick_success(self):
        """Test quick database check success"""
        with patch('app.api.v1.health.SupabaseDBHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler.health_check = AsyncMock(return_value=True)
            mock_handler_class.return_value = mock_handler
            
            result = asyncio.run(check_database_quick())
            
            assert result == True
            mock_handler.health_check.assert_called_once()
    
    def test_check_database_quick_failure(self):
        """Test quick database check failure"""
        with patch('app.api.v1.health.SupabaseDBHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler.health_check = AsyncMock(return_value=False)
            mock_handler_class.return_value = mock_handler
            
            result = asyncio.run(check_database_quick())
            
            assert result == False
    
    def test_check_database_quick_exception(self):
        """Test quick database check with exception"""
        with patch('app.api.v1.health.SupabaseDBHandler') as mock_handler_class:
            mock_handler_class.side_effect = Exception("Database connection failed")
            
            result = asyncio.run(check_database_quick())
            
            assert result == False
    
    def test_check_redis_quick_success(self):
        """Test quick Redis check success"""
        with patch('app.api.v1.health.redis_cache') as mock_redis:
            mock_redis.health_check = AsyncMock(return_value=True)
            
            result = asyncio.run(check_redis_quick())
            
            assert result == True
            mock_redis.health_check.assert_called_once()
    
    def test_check_redis_quick_failure(self):
        """Test quick Redis check failure"""
        with patch('app.api.v1.health.redis_cache') as mock_redis:
            mock_redis.health_check = AsyncMock(return_value=False)
            
            result = asyncio.run(check_redis_quick())
            
            assert result == False
    
    def test_check_redis_quick_exception(self):
        """Test quick Redis check with exception"""
        with patch('app.api.v1.health.redis_cache') as mock_redis:
            mock_redis.health_check = AsyncMock(side_effect=Exception("Redis connection failed"))
            
            result = asyncio.run(check_redis_quick())
            
            assert result == False


class TestHealthEndpointIntegration:
    """Integration tests for health endpoints"""
    
    def test_health_endpoints_response_structure(self):
        """Test all health endpoints return expected structure"""
        # Basic health check
        basic_result = asyncio.run(health_check())
        assert "status" in basic_result
        assert "timestamp" in basic_result
        assert "service" in basic_result
        
        # Liveness check
        liveness_result = asyncio.run(liveness_check())
        assert "status" in liveness_result
        assert "timestamp" in liveness_result
        assert liveness_result["status"] == "alive"
    
    def test_health_check_timing(self):
        """Test health check response timing"""
        start_time = datetime.utcnow()
        result = asyncio.run(health_check())
        end_time = datetime.utcnow()
        
        response_time = (end_time - start_time).total_seconds() * 1000
        
        # Health check should be fast (under 100ms for basic check)
        assert response_time < 100
        
        # Timestamp should be within the execution window
        result_time = datetime.fromisoformat(result["timestamp"])
        assert start_time <= result_time <= end_time
    
    def test_detailed_health_check_comprehensive(self):
        """Test detailed health check with comprehensive mocking"""
        mock_db_handler = MagicMock()
        mock_db_handler.health_check = AsyncMock(return_value=True)
        mock_db_handler.get_connection_stats = AsyncMock(return_value={
            "active_connections": 3,
            "pool_size": 10,
            "total_requests": 1000,
            "successful_requests": 995
        })
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.health_check = AsyncMock(return_value=True)
        mock_redis_cache.get_stats = AsyncMock(return_value={
            "connected_clients": 5,
            "used_memory": 2048000,
            "used_memory_human": "2.05M",
            "keyspace_hits": 150,
            "keyspace_misses": 25
        })
        
        result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
        
        # Verify overall structure
        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"
        assert isinstance(result["response_time_ms"], (int, float))
        
        # Verify database check details
        db_check = result["checks"]["database"]
        assert db_check["status"] == "healthy"
        assert db_check["stats"]["active_connections"] == 3
        assert db_check["stats"]["total_requests"] == 1000
        
        # Verify Redis check details
        redis_check = result["checks"]["redis"]
        assert redis_check["status"] == "healthy"
        assert redis_check["stats"]["connected_clients"] == 5
        assert redis_check["stats"]["keyspace_hits"] == 150
    
    def test_readiness_check_timing(self):
        """Test readiness check timing and parallel execution"""
        with patch('app.api.v1.health.check_database_quick', new_callable=AsyncMock) as mock_db_check:
            with patch('app.api.v1.health.check_redis_quick', new_callable=AsyncMock) as mock_redis_check:
                # Simulate some delay in checks
                async def delayed_db_check():
                    await asyncio.sleep(0.1)
                    return True
                
                async def delayed_redis_check():
                    await asyncio.sleep(0.1)
                    return True
                
                mock_db_check.side_effect = delayed_db_check
                mock_redis_check.side_effect = delayed_redis_check
                
                start_time = datetime.utcnow()
                result = asyncio.run(readiness_check())
                end_time = datetime.utcnow()
                
                # Should complete in ~100ms (parallel execution), not ~200ms (sequential)
                response_time = (end_time - start_time).total_seconds() * 1000
                assert response_time < 150  # Allow some overhead
                assert result["status"] == "ready"
    
    def test_health_check_error_scenarios(self):
        """Test various error scenarios across health checks"""
        # Test different combinations of service failures
        scenarios = [
            (True, False, "degraded"),    # DB healthy, Redis unhealthy
            (False, True, "degraded"),    # DB unhealthy, Redis healthy
            (False, False, "unhealthy"),  # Both unhealthy
        ]
        
        for db_healthy, redis_healthy, expected_status in scenarios:
            mock_db_handler = MagicMock()
            mock_db_handler.health_check = AsyncMock(return_value=db_healthy)
            if db_healthy:
                mock_db_handler.get_connection_stats = AsyncMock(return_value={})
            
            mock_redis_cache = MagicMock()
            mock_redis_cache.health_check = AsyncMock(return_value=redis_healthy)
            if redis_healthy:
                mock_redis_cache.get_stats = AsyncMock(return_value={})
            
            if expected_status == "unhealthy":
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["status"] == expected_status
            else:
                result = asyncio.run(detailed_health_check(mock_db_handler, mock_redis_cache))
                assert result["status"] == expected_status
