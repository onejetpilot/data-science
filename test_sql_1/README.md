# SQL: Агрегация нарушений ПДД и операций обработки

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-SQL-blue)
![SQL](https://img.shields.io/badge/SQL-EXISTS%20%7C%20JOIN-informational)

## О проекте

Мини-задача по SQL: по таблицам `viols` (нарушения ПДД) и `opers` (операции обработки) посчитать дневные агрегаты за период **2018-04-01 — 2018-04-20** (включительно) и сравнить два подхода: `LEFT JOIN` vs `EXISTS`.

## Данные

* `viols` - все зафиксированные нарушения.
* `opers` - операции по обработке нарушений (для части нарушений записей может не быть).

Поля:

* `viols.tr_viol_id`, `viols.time_check`, `viols.refuse_code`
* `opers.tr_viol_id`, `opers.oper_code`

Период задаётся так, чтобы включить весь день 2018-04-20:

```sql
WHERE time_check >= DATE '2018-04-01'
  AND time_check <  DATE '2018-04-21'
```

## Задача

Для каждой даты посчитать:

* `viols_total` - всего нарушений
* `refuse_0` - нарушений с `refuse_code = 0`
* `oper_29` - нарушений, по которым есть операция `oper_code = 29`
* `oper_no29_refuse_0` - нарушений с `refuse_code = 0`, по которым **нет** операции `oper_code = 29`

## Решение

### Вариант 1 - CTE + LEFT JOIN

```sql
WITH viol_base AS (
    SELECT
        tr_viol_id,
        time_check::date AS viol_date,
        refuse_code
    FROM viols
    WHERE time_check >= DATE '2018-04-01'
      AND time_check <  DATE '2018-04-21'
),
op29 AS (
    SELECT tr_viol_id
    FROM opers
    WHERE oper_code = 29
    GROUP BY tr_viol_id
)
SELECT
    vb.viol_date,
    COUNT(*) AS viols_total,
    COUNT(*) FILTER (WHERE vb.refuse_code = 0) AS refuse_0,
    COUNT(*) FILTER (WHERE o.tr_viol_id IS NOT NULL) AS oper_29,
    COUNT(*) FILTER (
        WHERE vb.refuse_code = 0
          AND o.tr_viol_id IS NULL
    ) AS oper_no29_refuse_0
FROM viol_base vb
LEFT JOIN op29 o
  ON o.tr_viol_id = vb.tr_viol_id
GROUP BY vb.viol_date
ORDER BY vb.viol_date;
```

### Вариант 2 - EXISTS / NOT EXISTS

```sql
SELECT
    v.time_check::date AS viol_date,
    COUNT(*) AS viols_total,
    COUNT(*) FILTER (WHERE v.refuse_code = 0) AS refuse_0,
    COUNT(*) FILTER (
        WHERE EXISTS (
            SELECT 1
            FROM opers o
            WHERE o.tr_viol_id = v.tr_viol_id
              AND o.oper_code = 29
        )
    ) AS oper_29,
    COUNT(*) FILTER (
        WHERE v.refuse_code = 0
          AND NOT EXISTS (
              SELECT 1
              FROM opers o
              WHERE o.tr_viol_id = v.tr_viol_id
                AND o.oper_code = 29
          )
    ) AS oper_no29_refuse_0
FROM viols v
WHERE v.time_check >= DATE '2018-04-01'
  AND v.time_check <  DATE '2018-04-21'
GROUP BY v.time_check::date
ORDER BY viol_date;
```

## Почему вариант 2 чаще быстрее

`EXISTS` — проверка наличия: базе достаточно найти первую подходящую строку в `opers` и остановиться. Вариант 1 сначала формирует отдельный набор `op29` (включая `GROUP BY`), потом делает `LEFT JOIN`, что обычно добавляет лишнюю работу.

Если нужен «максимальный буст» под вариант 2, полезен индекс:

```sql
CREATE INDEX IF NOT EXISTS idx_opers_tr_viol_id_oper_code
ON opers (tr_viol_id, oper_code);
```

## Вывод

Оба варианта корректны, но в задачах «проверить, есть ли связанная запись» чаще выгоднее использовать `EXISTS / NOT EXISTS`.

