"""
AI-Powered Video Commerce Recommender - Health Monitoring System
================================================================

This module provides comprehensive health monitoring for all system components,
performance metrics tracking, and alerting capabilities for production deployment.
"""

import asyncio
import time
import logging
import psutil
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

# Local imports
from models import HealthStatus, ComponentHealth, HealthResponse
from feature_store import FeatureStore
from content_processor import ContentProcessor
from recommender import RecommendationEngine
from ranking import RankingModel
from vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """System alert representation."""
    level: AlertLevel
    component: str
    message: str
    timestamp: float
    metadata: Dict[str, Any]

class SystemMetrics:
    """System resource and performance metrics collector."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.max_history_size = 1000
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            metrics = {
                'timestamp': time.time(),
                'uptime_seconds': time.time() - self.start_time,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percent': memory.percent
                },
                'swap': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'percent': swap.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
            if gpu_metrics:
                metrics['gpu'] = gpu_metrics
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if CUDA is available."""
        try:
            if not torch.cuda.is_available():
                return None
            
            gpu_count = torch.cuda.device_count()
            gpu_metrics = {
                'available': True,
                'device_count': gpu_count,
                'devices': []
            }
            
            for i in range(gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                
                device_info = {
                    'device_id': i,
                    'name': device_props.name,
                    'memory_total_gb': device_props.total_memory / (1024**3),
                    'memory_allocated_gb': memory_allocated,
                    'memory_cached_gb': memory_cached,
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
                gpu_metrics['devices'].append(device_info)
            
            return gpu_metrics
            
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated metrics summary for the specified time window."""
        try:
            current_time = time.time()
            window_start = current_time - (time_window_minutes * 60)
            
            # Filter metrics within time window
            window_metrics = [
                m for m in self.metrics_history 
                if m.get('timestamp', 0) >= window_start
            ]
            
            if not window_metrics:
                return {'error': 'No metrics available for time window'}
            
            # Calculate averages
            cpu_values = [m['cpu']['percent'] for m in window_metrics if 'cpu' in m]
            memory_values = [m['memory']['percent'] for m in window_metrics if 'memory' in m]
            disk_values = [m['disk']['percent'] for m in window_metrics if 'disk' in m]
            
            summary = {
                'time_window_minutes': time_window_minutes,
                'sample_count': len(window_metrics),
                'cpu_avg_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'cpu_max_percent': max(cpu_values) if cpu_values else 0,
                'memory_avg_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
                'memory_max_percent': max(memory_values) if memory_values else 0,
                'disk_avg_percent': sum(disk_values) / len(disk_values) if disk_values else 0,
                'current_uptime_hours': (current_time - self.start_time) / 3600
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: List[Alert] = []
        self.max_alerts = max_alerts
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0,
            'response_time_warning': 1000.0,  # ms
            'response_time_critical': 3000.0,  # ms
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.10,  # 10%
        }
    
    def add_alert(
        self, 
        level: AlertLevel, 
        component: str, 
        message: str, 
        metadata: Dict[str, Any] = None
    ):
        """Add a new alert."""
        alert = Alert(
            level=level,
            component=component,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Maintain max alerts limit
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.info,
            AlertLevel.WARNING: logging.warning,
            AlertLevel.ERROR: logging.error,
            AlertLevel.CRITICAL: logging.critical
        }.get(level, logging.info)
        
        log_level(f"ALERT [{level.upper()}] {component}: {message}")
    
    def check_system_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        try:
            # CPU alerts
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent > self.thresholds['cpu_critical']:
                self.add_alert(
                    AlertLevel.CRITICAL, 
                    'system', 
                    f"CPU usage critical: {cpu_percent:.1f}%",
                    {'cpu_percent': cpu_percent}
                )
            elif cpu_percent > self.thresholds['cpu_warning']:
                self.add_alert(
                    AlertLevel.WARNING, 
                    'system', 
                    f"CPU usage high: {cpu_percent:.1f}%",
                    {'cpu_percent': cpu_percent}
                )
            
            # Memory alerts
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            if memory_percent > self.thresholds['memory_critical']:
                self.add_alert(
                    AlertLevel.CRITICAL, 
                    'system', 
                    f"Memory usage critical: {memory_percent:.1f}%",
                    {'memory_percent': memory_percent}
                )
            elif memory_percent > self.thresholds['memory_warning']:
                self.add_alert(
                    AlertLevel.WARNING, 
                    'system', 
                    f"Memory usage high: {memory_percent:.1f}%",
                    {'memory_percent': memory_percent}
                )
            
            # Disk alerts
            disk_percent = metrics.get('disk', {}).get('percent', 0)
            if disk_percent > self.thresholds['disk_critical']:
                self.add_alert(
                    AlertLevel.CRITICAL, 
                    'system', 
                    f"Disk usage critical: {disk_percent:.1f}%",
                    {'disk_percent': disk_percent}
                )
            elif disk_percent > self.thresholds['disk_warning']:
                self.add_alert(
                    AlertLevel.WARNING, 
                    'system', 
                    f"Disk usage high: {disk_percent:.1f}%",
                    {'disk_percent': disk_percent}
                )
            
        except Exception as e:
            logger.error(f"Error checking system thresholds: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [
            {
                'level': alert.level,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'time_ago_hours': (time.time() - alert.timestamp) / 3600,
                'metadata': alert.metadata
            }
            for alert in self.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert counts by level and component."""
        summary = {
            'total_alerts': len(self.alerts),
            'by_level': {level.value: 0 for level in AlertLevel},
            'by_component': {},
            'recent_24h': 0
        }
        
        cutoff_24h = time.time() - (24 * 3600)
        
        for alert in self.alerts:
            # Count by level
            summary['by_level'][alert.level] += 1
            
            # Count by component
            if alert.component not in summary['by_component']:
                summary['by_component'][alert.component] = 0
            summary['by_component'][alert.component] += 1
            
            # Count recent alerts
            if alert.timestamp >= cutoff_24h:
                summary['recent_24h'] += 1
        
        return summary

class HealthChecker:
    """Main health checker that monitors all system components."""
    
    def __init__(
        self,
        feature_store: FeatureStore,
        content_processor: ContentProcessor,
        recommendation_engine: RecommendationEngine,
        ranking_model: RankingModel = None,
        vector_search: VectorSearchEngine = None
    ):
        self.feature_store = feature_store
        self.content_processor = content_processor
        self.recommendation_engine = recommendation_engine
        self.ranking_model = ranking_model
        self.vector_search = vector_search
        
        self.system_metrics = SystemMetrics()
        self.alert_manager = AlertManager()
        
        # Health check intervals (seconds)
        self.check_intervals = {
            'system_metrics': 60,      # Every minute
            'component_health': 300,   # Every 5 minutes
            'deep_health': 1800        # Every 30 minutes
        }
        
        self.last_checks = {
            'system_metrics': 0,
            'component_health': 0,
            'deep_health': 0
        }
        
        logger.info("Health checker initialized")
    
    async def check_system_health(self) -> HealthResponse:
        """Perform comprehensive system health check."""
        try:
            current_time = time.time()
            
            # Collect system metrics
            system_metrics = self.system_metrics.collect_system_metrics()
            self.alert_manager.check_system_thresholds(system_metrics)
            
            # Check individual components
            components = {}
            overall_status = HealthStatus.HEALTHY
            
            # Feature Store (Redis)
            components['feature_store'] = await self._check_feature_store()
            if components['feature_store'].status != HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
            
            # Content Processor
            components['content_processor'] = await self._check_content_processor()
            if components['content_processor'].status != HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
            
            # Recommendation Engine
            components['recommendation_engine'] = await self._check_recommendation_engine()
            if components['recommendation_engine'].status != HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
            
            # Ranking Model (if available)
            if self.ranking_model:
                components['ranking_model'] = await self._check_ranking_model()
                if components['ranking_model'].status != HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            # Vector Search (if available)
            if self.vector_search:
                components['vector_search'] = await self._check_vector_search()
                if components['vector_search'].status != HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            # System Resources
            components['system_resources'] = self._check_system_resources(system_metrics)
            if components['system_resources'].status != HealthStatus.HEALTHY:
                if components['system_resources'].status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            return HealthResponse(
                status=overall_status,
                components=components,
                timestamp=current_time,
                uptime_seconds=current_time - self.system_metrics.start_time
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status=HealthStatus.UNHEALTHY,
                components={},
                timestamp=time.time(),
                uptime_seconds=0
            )
    
    async def _check_feature_store(self) -> ComponentHealth:
        """Check feature store (Redis) health."""
        start_time = time.time()
        
        try:
            health_data = await self.feature_store.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_data.get('status') == 'healthy':
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time
                )
            else:
                error_msg = health_data.get('error', 'Feature store unhealthy')
                self.alert_manager.add_alert(
                    AlertLevel.ERROR, 
                    'feature_store', 
                    f"Feature store health check failed: {error_msg}"
                )
                
                return ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Feature store check failed: {e}"
            self.alert_manager.add_alert(AlertLevel.CRITICAL, 'feature_store', error_msg)
            
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg
            )
    
    async def _check_content_processor(self) -> ComponentHealth:
        """Check content processor health."""
        start_time = time.time()
        
        try:
            health_data = self.content_processor.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_data.get('initialized') and health_data.get('clip_model_loaded'):
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time
                )
            else:
                error_msg = "Content processor not fully initialized"
                self.alert_manager.add_alert(
                    AlertLevel.WARNING, 
                    'content_processor', 
                    error_msg
                )
                
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Content processor check failed: {e}"
            self.alert_manager.add_alert(AlertLevel.ERROR, 'content_processor', error_msg)
            
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg
            )
    
    async def _check_recommendation_engine(self) -> ComponentHealth:
        """Check recommendation engine health."""
        start_time = time.time()
        
        try:
            health_data = self.recommendation_engine.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_data.get('status') == 'healthy':
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time
                )
            else:
                error_msg = health_data.get('error', 'Recommendation engine unhealthy')
                self.alert_manager.add_alert(
                    AlertLevel.WARNING, 
                    'recommendation_engine', 
                    error_msg
                )
                
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Recommendation engine check failed: {e}"
            self.alert_manager.add_alert(AlertLevel.ERROR, 'recommendation_engine', error_msg)
            
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg
            )
    
    async def _check_ranking_model(self) -> ComponentHealth:
        """Check ranking model health."""
        start_time = time.time()
        
        try:
            health_data = self.ranking_model.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_data.get('status') == 'healthy':
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time
                )
            else:
                error_msg = health_data.get('error', 'Ranking model unhealthy')
                self.alert_manager.add_alert(
                    AlertLevel.WARNING, 
                    'ranking_model', 
                    error_msg
                )
                
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Ranking model check failed: {e}"
            self.alert_manager.add_alert(AlertLevel.ERROR, 'ranking_model', error_msg)
            
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg
            )
    
    async def _check_vector_search(self) -> ComponentHealth:
        """Check vector search engine health."""
        start_time = time.time()
        
        try:
            health_data = self.vector_search.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_data.get('status') == 'healthy':
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time
                )
            else:
                error_msg = health_data.get('error', 'Vector search unhealthy')
                self.alert_manager.add_alert(
                    AlertLevel.WARNING, 
                    'vector_search', 
                    error_msg
                )
                
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Vector search check failed: {e}"
            self.alert_manager.add_alert(AlertLevel.ERROR, 'vector_search', error_msg)
            
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg
            )
    
    def _check_system_resources(self, metrics: Dict[str, Any]) -> ComponentHealth:
        """Check system resource health."""
        try:
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check CPU
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            # Check Memory
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Memory usage high: {memory_percent:.1f}%")
            
            # Check Disk
            disk_percent = metrics.get('disk', {}).get('percent', 0)
            if disk_percent > 98:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            error_message = "; ".join(issues) if issues else None
            
            return ComponentHealth(
                status=status,
                response_time_ms=0,
                error_message=error_message
            )
            
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error_message=f"System resource check failed: {e}"
            )
    
    async def periodic_health_check(self):
        """Run periodic health checks in background."""
        try:
            while True:
                current_time = time.time()
                
                # System metrics check
                if current_time - self.last_checks['system_metrics'] > self.check_intervals['system_metrics']:
                    metrics = self.system_metrics.collect_system_metrics()
                    self.alert_manager.check_system_thresholds(metrics)
                    self.last_checks['system_metrics'] = current_time
                
                # Component health check
                if current_time - self.last_checks['component_health'] > self.check_intervals['component_health']:
                    health_response = await self.check_system_health()
                    
                    # Generate alerts for unhealthy components
                    if health_response.status != HealthStatus.HEALTHY:
                        unhealthy_components = [
                            name for name, health in health_response.components.items()
                            if health.status == HealthStatus.UNHEALTHY
                        ]
                        
                        if unhealthy_components:
                            self.alert_manager.add_alert(
                                AlertLevel.ERROR,
                                'system',
                                f"Unhealthy components detected: {', '.join(unhealthy_components)}"
                            )
                    
                    self.last_checks['component_health'] = current_time
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Periodic health check cancelled")
        except Exception as e:
            logger.error(f"Error in periodic health check: {e}")
    
    def get_system_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive system status summary."""
        try:
            # Get recent metrics
            metrics_summary = self.system_metrics.get_metrics_summary(60)  # Last hour
            
            # Get alerts
            alert_summary = self.alert_manager.get_alert_summary()
            recent_alerts = self.alert_manager.get_recent_alerts(24)  # Last 24 hours
            
            return {
                'timestamp': time.time(),
                'uptime_hours': (time.time() - self.system_metrics.start_time) / 3600,
                'system_metrics': metrics_summary,
                'alerts': {
                    'summary': alert_summary,
                    'recent': recent_alerts[:10]  # Most recent 10 alerts
                },
                'last_health_check': max(self.last_checks.values()),
                'health_check_intervals': self.check_intervals
            }
            
        except Exception as e:
            logger.error(f"Error getting system status summary: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def start_monitoring(self):
        """Start background monitoring tasks."""
        try:
            # Start periodic health checks
            task = asyncio.create_task(self.periodic_health_check())
            logger.info("Health monitoring started")
            return task
            
        except Exception as e:
            logger.error(f"Error starting health monitoring: {e}")
            return None