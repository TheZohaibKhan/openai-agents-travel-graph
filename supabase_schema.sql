-- OPTIONAL. Only needed if you turn on SAVE_TO_SUPABASE=true in .env.
-- Paste this into Supabase -> SQL Editor -> Run.

create table if not exists public.trip_plans (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  destination text not null,
  request jsonb not null,
  plan jsonb not null
);

alter table public.trip_plans enable row level security;
-- No policies are created on purpose: with Row Level Security on and no policy,
-- only the secret "service_role" key can read/write this table. Keep that key
-- ONLY in your local .env file and never share or commit it.
