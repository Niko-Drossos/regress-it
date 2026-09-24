-- 002_run_telemetry.sql
-- Adds stopping-criterion and loss-telemetry columns to runs.
-- Apply AFTER 001_init.sql: SQL Editor -> New query -> paste -> Run.
--
-- Why:
--   * status / epochs_run record which stopping rule ended the run, so the
--     Run History tab can show converged vs. diverged side by side.
--   * patience / min_delta record the early-stopping config used, so two runs
--     are only compared when their stopping rules match.
--   * loss_history stores the per-epoch training loss, so any past run's loss
--     curve can be redrawn from GET /runs/{run_id}.
--   * mse / mae / r2 become nullable: a diverged run can produce NaN/inf
--     metrics, which JSON cannot carry. The API stores those as NULL instead of
--     failing the insert, so diverged runs are still recorded.

alter table runs
    add column if not exists status       text    not null default 'max_epochs',
    add column if not exists epochs_run   integer,
    add column if not exists patience     integer,
    add column if not exists min_delta    double precision,
    add column if not exists loss_history jsonb   not null default '[]'::jsonb,
    alter column mse drop not null,
    alter column mae drop not null,
    alter column r2  drop not null;

alter table runs drop constraint if exists runs_status_check;
alter table runs
    add constraint runs_status_check
    check (status in ('converged', 'max_epochs', 'diverged'));

-- The existing "anon can read runs" policy covers the new columns; no RLS change.
