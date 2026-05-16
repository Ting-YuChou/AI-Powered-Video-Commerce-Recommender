{{/*
Expand the chart name.
*/}}
{{- define "video-commerce.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "video-commerce.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create a component resource name.
*/}}
{{- define "video-commerce.componentName" -}}
{{- $root := .root -}}
{{- $name := .name -}}
{{- printf "%s-%s" (include "video-commerce.fullname" $root) $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Chart label.
*/}}
{{- define "video-commerce.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "video-commerce.labels" -}}
helm.sh/chart: {{ include "video-commerce.chart" . }}
app.kubernetes.io/name: {{ include "video-commerce.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
Selector labels for a single component.
*/}}
{{- define "video-commerce.selectorLabels" -}}
app.kubernetes.io/name: {{ include "video-commerce.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Service account name.
*/}}
{{- define "video-commerce.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "video-commerce.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Secret name.
*/}}
{{- define "video-commerce.secretName" -}}
{{- default (printf "%s-secrets" (include "video-commerce.fullname" .)) .Values.secrets.existingSecret -}}
{{- end -}}

{{/*
Backend image reference.
*/}}
{{- define "video-commerce.backendImage" -}}
{{- printf "%s:%s" .Values.images.backend.repository .Values.images.backend.tag -}}
{{- end -}}

{{/*
Frontend/Caddy image reference.
*/}}
{{- define "video-commerce.frontendImage" -}}
{{- printf "%s:%s" .Values.images.frontend.repository .Values.images.frontend.tag -}}
{{- end -}}

{{/*
Common secret environment variables.
*/}}
{{- define "video-commerce.secretEnv" -}}
- name: API_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.apiApiKey }}
      optional: true
- name: SECURITY_INTERNAL_SERVICE_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.internalServiceKey }}
      optional: true
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.redisPassword }}
      optional: true
- name: REDIS_CACHE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.redisCachePassword }}
      optional: true
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.databaseUrl }}
      optional: true
- name: OBJECT_STORAGE_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.objectStorageAccessKeyId }}
      optional: true
- name: OBJECT_STORAGE_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.objectStorageSecretAccessKey }}
      optional: true
- name: SECURITY_JWT_SHARED_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "video-commerce.secretName" . }}
      key: {{ .Values.secrets.keys.jwtSharedSecret }}
      optional: true
{{- end -}}

{{/*
Render a map as simple quoted env var entries.
*/}}
{{- define "video-commerce.mapEnv" -}}
{{- range $name, $value := . }}
- name: {{ $name }}
  value: {{ $value | toString | quote }}
{{- end }}
{{- end -}}

{{/*
Fail when the chart is configured with local object storage.
*/}}
{{- define "video-commerce.requireS3ObjectStorage" -}}
{{- if ne .Values.external.objectStorage.backend "s3" -}}
{{- fail "Kubernetes multi-node deployments require external.objectStorage.backend=s3" -}}
{{- end -}}
{{- end -}}
