package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Duration;
import org.junit.jupiter.api.Test;

class IcebergMaintenanceJobTest {
  @Test
  void maintenanceDefaultsAreConservativeAndBounded() {
    IcebergMaintenanceJob.Policy policy = IcebergMaintenanceJob.Policy.defaults();

    assertEquals(Duration.ofDays(30), policy.maxSnapshotAge);
    assertEquals(20, policy.retainLast);
    assertEquals(Duration.ofDays(7), policy.orphanMinAge);
    assertEquals(1000, policy.deleteBatchSize);
  }

  @Test
  void maintenanceOnlyAcceptsExplicitTableAllowlist() {
    assertEquals(
        "ranking_training_pit",
        IcebergMaintenanceJob.validateTable("ranking_training_pit"));
    assertThrows(
        IllegalArgumentException.class,
        () -> IcebergMaintenanceJob.validateTable("some_other_table"));
  }

  @Test
  void maintenanceRejectsAHeadSnapshotChangedAfterPlanning() {
    IcebergMaintenanceJob.validateHeadSnapshot(42L, 42L);
    assertThrows(
        IllegalStateException.class,
        () -> IcebergMaintenanceJob.validateHeadSnapshot(43L, 42L));
  }
}
