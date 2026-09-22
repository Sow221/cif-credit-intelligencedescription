# Terraform — Provisionnement de l'infrastructure CIF sur Oracle Cloud Free Tier
# (Always-Free: VM.Standard.A1.Flex ARM 4 OCPU / 24 Go + K3s).
#
# Périmètre de ce fichier : provisionner la VM et installer K3s dessus. Rien de plus —
# la création des secrets et l'application des manifestes k8s/*.yaml sont faites une seule
# fois, par `scripts/deploy.sh` (via SSH + kubectl, après provisionnement), pas ici. Les faire
# aussi dans le cloud-init dupliquerait le travail et créerait un secret incomplet au premier
# démarrage (client_secret manquant), avant que deploy.sh ne le corrige — source de confusion
# évitée en gardant une seule source de vérité pour la couche applicative.
#
# Usage:
#   terraform init
#   terraform apply \
#     -var="tenancy_ocid=ocid1.tenancy.." \
#     -var="user_ocid=ocid1.user.." \
#     -var="compartment_ocid=ocid1.compartment.." \
#     -var="fingerprint=xx:xx" \
#     -var="private_key_path=~/.oci/key.pem" \
#     -var="ssh_public_key=ssh-rsa AAAA..."

terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0"
    }
  }
}

variable "tenancy_ocid" {
  type = string
}

variable "user_ocid" {
  type = string
}

variable "compartment_ocid" {
  type = string
}

variable "fingerprint" {
  type = string
}

variable "private_key_path" {
  type = string
}

variable "ssh_public_key" {
  type = string
}

variable "region" {
  type    = string
  default = "eu-marseille-1"
}

variable "instance_shape" {
  type    = string
  default = "VM.Standard.A1.Flex"
}

variable "instance_ocpus" {
  type    = number
  default = 4
}

variable "instance_memory" {
  type    = number
  default = 24
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.instance_shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_vcn" "cif_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "cif-vcn"
  dns_label      = "cifvcn"
}

resource "oci_core_subnet" "cif_subnet" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.cif_vcn.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "cif-subnet"
  dns_label         = "cifsub"
  security_list_ids = [oci_core_security_list.cif_sl.id]
}

resource "oci_core_security_list" "cif_sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.cif_vcn.id
  display_name   = "cif-sl"

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 6443
      max = 6443
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 8000
      max = 8000
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 30080
      max = 30080
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

resource "oci_core_instance" "cif_k3s" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "cif-k3s"
  shape               = var.instance_shape

  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu.images[0].id
  }

  create_vnic_details {
    assign_public_ip = true
    subnet_id        = oci_core_subnet.cif_subnet.id
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(<<-EOT
      #cloud-config
      runcmd:
        - curl -sfL https://get.k3s.io | sh -
        - sleep 20
        - chmod 644 /etc/rancher/k3s/k3s.yaml
    EOT
    )
  }
}

output "instance_public_ip" {
  value = oci_core_instance.cif_k3s.public_ip
}
